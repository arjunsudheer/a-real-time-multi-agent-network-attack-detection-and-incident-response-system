import json
import subprocess
import shlex
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd


class MitigationCommand(BaseModel):
    command_type: str = Field(
        description="Type of mitigation command (e.g., 'block_host', 'block_port', 'rate_limit')",
        min_length=3,
        max_length=50,
    )
    description: str = Field(
        description="Human-readable description of what this command does",
        min_length=10,
        max_length=200,
    )
    curl_command: str = Field(
        description="Complete curl command to execute the mitigation",
        min_length=50,
    )
    priority: int = Field(
        description="Flow rule priority (higher number = higher priority)",
        ge=1,
        le=65535,
        default=100,
    )
    dpid: int = Field(
        description="Datapath ID of the switch to configure",
        ge=1,
        default=1,
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "command_type": "block_host",
                    "description": "Block all traffic from malicious host 10.0.0.1",
                    "curl_command": 'curl -X POST http://localhost:8080/stats/flowentry/add -d \'{"dpid": 1, "priority": 100, "match": {"ipv4_src": "10.0.0.1", "eth_type": 2048}, "actions": []}\'',
                    "priority": 100,
                    "dpid": 1,
                }
            ]
        }


class NetworkNode(BaseModel):
    id: str = Field(description="Unique identifier for the node")
    type: str = Field(description="Type of node: 'switch', 'host', 'controller'")
    label: str = Field(description="Human-readable label")
    status: str = Field(
        description="Status: 'active', 'blocked', 'inactive'", default="active"
    )
    ip_address: Optional[str] = Field(
        description="IP address if applicable", default=None
    )
    dpid: Optional[int] = Field(description="Datapath ID for switches", default=None)
    port_count: int = Field(description="Number of ports", default=0)
    flow_count: int = Field(description="Number of flow entries", default=0)
    blocking_commands: List[str] = Field(
        description="Commands used to block this node", default=[]
    )


class NetworkLink(BaseModel):
    id: str = Field(description="Unique identifier for the link")
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    source_port: Optional[int] = Field(description="Source port number", default=None)
    target_port: Optional[int] = Field(description="Target port number", default=None)
    status: str = Field(
        description="Status: 'active', 'blocked', 'inactive'", default="active"
    )
    bandwidth: Optional[str] = Field(description="Link bandwidth", default=None)


class NetworkTopology(BaseModel):
    nodes: List[NetworkNode] = Field(description="Network nodes (switches, hosts)")
    links: List[NetworkLink] = Field(description="Network links")
    blocked_flows: List[Dict[str, Any]] = Field(
        description="Blocked flow entries", default=[]
    )
    mitigation_summary: str = Field(
        description="Summary of applied mitigations", default=""
    )
    last_updated: str = Field(description="Timestamp of last update")


class ResponseAgent:
    def __init__(self, ryu_host: str = "localhost:8080"):
        # IP to host mapping
        self.ip_to_host = {
            "10.0.0.1": "h1",
            "10.0.0.2": "h2",
            "10.0.0.3": "h3",
            "10.0.0.4": "h4",
            "10.0.0.5": "h5",
        }

        # Protocol mappings
        self.protocol_to_number = {"tcp": 6, "udp": 17, "icmp": 1}

        # Ryu controller configuration
        self.ryu_host = ryu_host
        self.ryu_base_url = f"http://{ryu_host}"

        # Track blocked entities and commands
        self.blocked_hosts = set()
        self.blocked_flows = []
        self.applied_commands = []

    def query_ryu_topology(self) -> Dict[str, List]:
        """Query the current network topology from Ryu controller"""
        try:
            topology_data = {
                "switches": [],
                "hosts": [],
                "links": [],
                "host_port_mapping": {},
            }

            # Get switches using correct Ryu REST API endpoint
            switches_url = f"{self.ryu_base_url}/stats/switches"
            print(f"Querying switches from: {switches_url}")
            response = requests.get(switches_url, timeout=5)

            if response.status_code == 200:
                switches = response.json()
                topology_data["switches"] = (
                    switches if isinstance(switches, list) else []
                )
                print(
                    f"Found {len(topology_data['switches'])} switches: {topology_data['switches']}"
                )
            else:
                print(f"Failed to get switches. Status: {response.status_code}")

            # Get port descriptions to understand network layout
            for dpid in topology_data["switches"]:
                try:
                    ports_url = f"{self.ryu_base_url}/stats/portdesc/{dpid}"
                    ports_response = requests.get(ports_url, timeout=5)

                    if ports_response.status_code == 200:
                        ports_data = ports_response.json()
                        ports = ports_data.get(str(dpid), [])

                        # Analyze ports to infer hosts
                        for port in ports:
                            port_no = port.get("port_no")
                            port_name = port.get("name", "")

                            # Skip internal switch ports (typically named s1-eth2, s1-eth3 for inter-switch links)
                            # Host ports are typically s1-eth1, s2-eth1, etc. (eth1 = first port = host port)
                            if port_no and port_no != 4294967294:  # Skip LOCAL port
                                if "eth1" in port_name or (
                                    port_no == 1
                                ):  # First port typically connects to host
                                    # Infer host based on switch and port
                                    host_ip = self._infer_host_ip_from_switch_port(
                                        dpid, port_no
                                    )
                                    if host_ip:
                                        topology_data["hosts"].append(
                                            {
                                                "dpid": dpid,
                                                "port": port_no,
                                                "ipv4": host_ip,
                                                "mac": "00:00:00:00:00:0"
                                                + str(len(topology_data["hosts"]) + 1),
                                            }
                                        )
                                        topology_data["host_port_mapping"][host_ip] = {
                                            "dpid": dpid,
                                            "port": port_no,
                                        }

                        print(f"Switch {dpid} has {len(ports)} ports")

                except Exception as e:
                    print(f"Error getting ports for switch {dpid}: {str(e)}")
                    continue

            # For testing purposes, if no hosts discovered, add some default ones based on common topology
            if not topology_data["hosts"] and topology_data["switches"]:
                print(
                    "No hosts discovered from ports, using fallback host detection..."
                )
                default_hosts = [
                    {
                        "dpid": 1,
                        "port": 1,
                        "ipv4": "10.0.0.1",
                        "mac": "00:00:00:00:00:01",
                    },
                    {
                        "dpid": 2,
                        "port": 1,
                        "ipv4": "10.0.0.2",
                        "mac": "00:00:00:00:00:02",
                    },
                    {
                        "dpid": 3,
                        "port": 1,
                        "ipv4": "10.0.0.3",
                        "mac": "00:00:00:00:00:03",
                    },
                ]

                for host in default_hosts:
                    if host["dpid"] in topology_data["switches"]:
                        topology_data["hosts"].append(host)
                        topology_data["host_port_mapping"][host["ipv4"]] = {
                            "dpid": host["dpid"],
                            "port": host["port"],
                        }

            print(
                f"Discovered {len(topology_data['hosts'])} hosts: {[h['ipv4'] for h in topology_data['hosts']]}"
            )
            return topology_data

        except Exception as e:
            print(f"Error querying Ryu topology: {str(e)}")
            return {"switches": [], "hosts": [], "links": [], "host_port_mapping": {}}

    def _infer_host_ip_from_switch_port(self, dpid: int, port_no: int) -> str:
        """Infer host IP based on switch ID and port number"""
        # Common patterns in mininet topologies
        if dpid == 1 and port_no == 1:
            return "10.0.0.1"
        elif dpid == 2 and port_no == 1:
            return "10.0.0.2"
        elif dpid == 3 and port_no == 1:
            return "10.0.0.3"
        else:
            # Generic pattern: 10.0.{switch_id}.{port_no}
            return f"10.0.{dpid}.{port_no}"

    def create_network_topology_graph(
        self, include_blocked: bool = True
    ) -> NetworkTopology:
        """Create a graph data structure representing the network topology"""
        from datetime import datetime

        # Query current topology from Ryu
        topology_data = self.query_ryu_topology()

        nodes = []
        links = []

        # Create switch nodes
        for dpid in topology_data["switches"]:
            try:
                # Get switch details
                response = requests.get(
                    f"{self.ryu_base_url}/stats/desc/{dpid}", timeout=5
                )
                switch_desc = (
                    response.json().get(str(dpid), {})
                    if response.status_code == 200
                    else {}
                )

                switch_node = NetworkNode(
                    id=f"switch_{dpid}",
                    type="switch",
                    label=f"Switch {dpid}",
                    status="active",
                    dpid=dpid,
                )
                nodes.append(switch_node)
            except Exception as e:
                print(f"Error getting switch {dpid} details: {str(e)}")
                # Create basic switch node even if details fail
                switch_node = NetworkNode(
                    id=f"switch_{dpid}",
                    type="switch",
                    label=f"Switch {dpid}",
                    status="active",
                    dpid=dpid,
                )
                nodes.append(switch_node)

        # Create host nodes from discovered hosts
        for host in topology_data["hosts"]:
            host_ip = host["ipv4"]
            host_dpid = host["dpid"]
            host_port = host["port"]

            # Determine if host is blocked
            host_status = "blocked" if host_ip in self.blocked_hosts else "active"

            host_node = NetworkNode(
                id=f"host_{host_ip.replace('.', '_')}",
                type="host",
                label=f"Host {host_ip}",
                status=host_status,
                ip_address=host_ip,
            )
            nodes.append(host_node)

            # Create link between host and switch
            host_switch_link = NetworkLink(
                id=f"link_host_{host_ip.replace('.', '_')}_switch_{host_dpid}",
                source=f"host_{host_ip.replace('.', '_')}",
                target=f"switch_{host_dpid}",
                source_port=None,
                target_port=host_port,
                status="active",
            )
            links.append(host_switch_link)

        # Create inter-switch links (for linear topology: s1-s2, s2-s3)
        switches = sorted(topology_data["switches"])
        for i in range(len(switches) - 1):
            switch1_dpid = switches[i]
            switch2_dpid = switches[i + 1]

            inter_switch_link = NetworkLink(
                id=f"link_switch_{switch1_dpid}_switch_{switch2_dpid}",
                source=f"switch_{switch1_dpid}",
                target=f"switch_{switch2_dpid}",
                source_port=None,
                target_port=None,
                status="active",
            )
            links.append(inter_switch_link)

        # Create mitigation summary
        mitigation_summary = (
            f"Applied {len(self.applied_commands)} mitigation commands. "
        )
        if self.blocked_hosts:
            blocked_host_names = [
                self._ip_to_host_name(ip) for ip in self.blocked_hosts
            ]
            mitigation_summary += f"Blocked hosts: {', '.join(blocked_host_names)}"
        else:
            mitigation_summary += "No hosts currently blocked."

        return NetworkTopology(
            nodes=nodes,
            links=links,
            blocked_flows=self.blocked_flows,
            mitigation_summary=mitigation_summary,
            last_updated=datetime.now().isoformat(),
        )

    def _ip_to_host_name(self, ip: str) -> str:
        """Convert IP address to host name (h1, h2, h3, etc.)"""
        # Use the existing mapping if available
        if ip in self.ip_to_host:
            return self.ip_to_host[ip]

        # Otherwise generate based on the last octet
        try:
            last_octet = int(ip.split(".")[-1])
            return f"h{last_octet}"
        except:
            return f"host_{ip.replace('.', '_')}"

    def generate_and_run_mitigation_commands(
        self,
        src_ip: str,
        dst_ip: str = None,
        protocol: str = "tcp",
        dst_port: int = None,
        dpid: int = 1,
        execute: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate and execute mitigation commands based on network parameters

        Args:
            src_ip: Source IP address to block
            dst_ip: Destination IP address (optional)
            protocol: Protocol type (tcp, udp, icmp)
            dst_port: Destination port to block
            dpid: Datapath ID of the switch
            execute: Whether to execute the commands

        Returns:
            Dictionary with generated commands and execution results
        """
        try:
            # ANSI color codes for green output
            GREEN = "\033[92m"
            BOLD = "\033[1m"
            RESET = "\033[0m"

            print(f"\n{GREEN}{BOLD}🛡️  Generating mitigation commands:{RESET}")
            print(
                f"{GREEN}Source IP: {src_ip} ({self.ip_to_host.get(src_ip, 'unknown host')}){RESET}"
            )
            print(f"{GREEN}Destination IP: {dst_ip}{RESET}")
            print(f"{GREEN}Protocol: {protocol}{RESET}")
            print(f"{GREEN}Port: {dst_port}{RESET}")

            commands = []

            # Generate appropriate blocking commands
            if protocol.lower() == "icmp":
                commands.append(self._create_icmp_block_command(src_ip, dpid))
            elif protocol.lower() in ["tcp", "udp"] and dst_port:
                commands.append(
                    self._create_port_block_command(src_ip, protocol, dst_port, dpid)
                )
            else:
                # Default: block all traffic from source IP
                commands.append(self._create_host_block_command(src_ip, dpid))

            result = {
                "commands": commands,
                "summary": self._get_command_summary(commands),
                "execution_results": None,
            }

            # Execute commands if requested
            if execute and commands:
                execution_results = self._execute_commands(commands)
                result["execution_results"] = execution_results

                # Track blocked hosts and commands
                self.blocked_hosts.add(src_ip)
                self.applied_commands.extend(commands)

            return result

        except Exception as e:
            print(f"Error generating mitigation commands: {str(e)}")
            return {
                "commands": [],
                "summary": "Error generating commands",
                "execution_results": None,
            }

    def _create_host_block_command(self, src_ip: str, dpid: int) -> MitigationCommand:
        """Block all traffic from a host"""
        host_name = self.ip_to_host.get(src_ip, "unknown")
        return MitigationCommand(
            command_type="block_host",
            description=f"Block all traffic from host {host_name} ({src_ip})",
            curl_command=f'curl -X POST http://localhost:8080/stats/flowentry/add -d \'{{"dpid": {dpid}, "priority": 100, "match": {{"ipv4_src": "{src_ip}", "eth_type": 2048}}, "actions": []}}\'',
            priority=100,
            dpid=dpid,
        )

    def _create_icmp_block_command(self, src_ip: str, dpid: int) -> MitigationCommand:
        """Block ICMP (ping) from a host"""
        host_name = self.ip_to_host.get(src_ip, "unknown")
        return MitigationCommand(
            command_type="block_icmp",
            description=f"Block ICMP (ping) from {host_name} ({src_ip})",
            curl_command=f'curl -X POST http://localhost:8080/stats/flowentry/add -d \'{{"dpid": {dpid}, "priority": 100, "match": {{"eth_type": 2048, "ip_proto": 1, "ipv4_src": "{src_ip}"}}, "actions": []}}\'',
            priority=100,
            dpid=dpid,
        )

    def _create_port_block_command(
        self, src_ip: str, protocol: str, dst_port: int, dpid: int
    ) -> MitigationCommand:
        """Block specific protocol/port traffic from a host"""
        host_name = self.ip_to_host.get(src_ip, "unknown")
        protocol_num = self.protocol_to_number.get(protocol.lower(), 6)

        if protocol.lower() == "tcp":
            match_dict = {
                "eth_type": 2048,
                "ip_proto": protocol_num,
                "tcp_dst": dst_port,
                "ipv4_src": src_ip,
            }
            description = (
                f"Block TCP traffic to port {dst_port} from {host_name} ({src_ip})"
            )
        elif protocol.lower() == "udp":
            match_dict = {
                "eth_type": 2048,
                "ip_proto": protocol_num,
                "udp_dst": dst_port,
                "ipv4_src": src_ip,
            }
            description = (
                f"Block UDP traffic to port {dst_port} from {host_name} ({src_ip})"
            )
        else:
            # Fallback to general block
            return self._create_host_block_command(src_ip, dpid)

        return MitigationCommand(
            command_type=f"block_{protocol.lower()}_port",
            description=description,
            curl_command=f'curl -X POST http://localhost:8080/stats/flowentry/add -d \'{{"dpid": {dpid}, "priority": 200, "match": {json.dumps(match_dict)}, "actions": []}}\'',
            priority=200,
            dpid=dpid,
        )

    def _execute_commands(self, commands: List[MitigationCommand]) -> Dict[str, Any]:
        """Execute the generated mitigation commands"""
        results = {
            "executed_commands": [],
            "failed_commands": [],
            "total_commands": len(commands),
            "success_count": 0,
        }

        for cmd in commands:
            try:
                # ANSI color codes for green output
                GREEN = "\033[92m"
                BOLD = "\033[1m"
                RESET = "\033[0m"

                print(f"\n{GREEN}{BOLD}⚡ Executing: {cmd.description}{RESET}")
                print(f"{GREEN}Command: {cmd.curl_command}{RESET}")

                # Parse and execute the curl command safely
                cmd_args = shlex.split(cmd.curl_command)
                result = subprocess.run(
                    cmd_args, capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    results["executed_commands"].append(
                        {
                            "command": cmd.description,
                            "curl_command": cmd.curl_command,
                            "output": result.stdout,
                            "status": "success",
                        }
                    )
                    results["success_count"] += 1
                    print(f"{GREEN}✅ Successfully executed: {cmd.description}{RESET}")
                else:
                    results["failed_commands"].append(
                        {
                            "command": cmd.description,
                            "curl_command": cmd.curl_command,
                            "error": result.stderr,
                            "status": "failed",
                        }
                    )
                    RED = "\033[91m"
                    print(f"{RED}❌ Failed to execute: {cmd.description}{RESET}")
                    print(f"{RED}Error: {result.stderr}{RESET}")

            except Exception as e:
                results["failed_commands"].append(
                    {
                        "command": cmd.description,
                        "curl_command": cmd.curl_command,
                        "error": str(e),
                        "status": "error",
                    }
                )
                RED = "\033[91m"
                RESET = "\033[0m"
                print(f"{RED}❌ Exception executing {cmd.description}: {str(e)}{RESET}")

        return results

    def _get_command_summary(self, commands: List[MitigationCommand]) -> str:
        """Generate a human-readable summary of mitigation actions"""
        if not commands:
            return "No mitigation commands generated."

        summary = f"Generated {len(commands)} mitigation commands:\n"
        for i, cmd in enumerate(commands, 1):
            summary += f"{i}. {cmd.description} (Priority: {cmd.priority})\n"

        return summary

    def generate_mitigation_commands(
        self,
        classification_results: Dict[str, Any],
        original_sample: pd.DataFrame,
        dpid: int = 1,
    ) -> List[MitigationCommand]:
        """
        Backward compatibility method that extracts parameters from classification results
        and original sample, then calls the main mitigation method
        """
        if classification_results.get("final_prediction", "Unknown") == "Benign":
            print("No mitigation needed for benign traffic")
            return []

        # Extract network information from the sample
        src_ip = (
            original_sample.get("Src IP", {}).iloc[0]
            if not original_sample.empty
            else "10.0.0.1"
        )
        dst_port = (
            original_sample.get("Dst Port", {}).iloc[0]
            if not original_sample.empty
            else None
        )
        protocol_num = (
            original_sample.get("Protocol", {}).iloc[0]
            if not original_sample.empty
            else None
        )

        # Convert protocol number to string
        protocol = "tcp"  # default
        if protocol_num == 17:
            protocol = "udp"
        elif protocol_num == 1:
            protocol = "icmp"
        elif protocol_num == 6:
            protocol = "tcp"

        # Generate and return just the commands (not execution results)
        result = self.generate_and_run_mitigation_commands(
            src_ip=str(src_ip),
            protocol=protocol,
            dst_port=int(dst_port) if dst_port and str(dst_port).isdigit() else None,
            dpid=dpid,
            execute=False,  # Don't execute, just generate
        )

        return result["commands"]

    def execute_mitigation_commands(
        self, commands: List[MitigationCommand]
    ) -> Dict[str, Any]:
        """Execute a list of mitigation commands"""
        # Track the commands when executed
        self.applied_commands.extend(commands)
        for cmd in commands:
            # Extract source IP from curl command to track blocked hosts
            if '"ipv4_src"' in cmd.curl_command:
                import re

                ip_match = re.search(r'"ipv4_src": "([^"]+)"', cmd.curl_command)
                if ip_match:
                    self.blocked_hosts.add(ip_match.group(1))

        return self._execute_commands(commands)

    def get_mitigation_summary(self, commands: List[MitigationCommand]) -> str:
        """Get summary of mitigation commands"""
        return self._get_command_summary(commands)

    def get_topology_for_ui(self) -> Dict[str, Any]:
        """Get network topology data formatted for UI consumption"""
        topology = self.create_network_topology_graph()

        # Convert to dict format suitable for JSON serialization
        return {
            "nodes": [node.dict() for node in topology.nodes],
            "links": [link.dict() for link in topology.links],
            "blocked_flows": topology.blocked_flows,
            "mitigation_summary": topology.mitigation_summary,
            "last_updated": topology.last_updated,
            "applied_commands": [
                {
                    "command_type": cmd.command_type,
                    "description": cmd.description,
                    "curl_command": cmd.curl_command,
                    "priority": cmd.priority,
                    "dpid": cmd.dpid,
                }
                for cmd in self.applied_commands
            ],
            "stats": {
                "total_nodes": len(topology.nodes),
                "total_links": len(topology.links),
                "blocked_hosts": len(self.blocked_hosts),
                "applied_commands": len(self.applied_commands),
            },
        }

    def reset_blocking_state(self):
        """Reset all tracking of blocked hosts and commands (for testing)"""
        self.blocked_hosts.clear()
        self.blocked_flows.clear()
        self.applied_commands.clear()
        print("Reset all blocking state")
