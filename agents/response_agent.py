import json
import subprocess
import shlex
import requests
import time
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from agents.knowledge_source import KnowledgeSource
from agents.llm_tools import safe_web_search_tool, safe_arxiv_retrieve_tool


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
    def __init__(
        self, ryu_host: str = "localhost:8080", use_long_term_memory: bool = False
    ):
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

        # RAG components
        self.use_long_term_memory = use_long_term_memory

        # Long term memory for mitigation strategies
        self.ltm_db = KnowledgeSource(
            Path("agents/response_agent_long_term_memory"),
        )

        self.__initialize_tools()
        self.__initialize_llm()

    def __initialize_llm(self) -> None:
        """
        Initialize the response agent using the Gemini 2.0 Flash model.
        Creates a ReAct agent with access to tools for intelligent mitigation generation.
        """
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyC72eGdAEHU9ZBAhXJWAg6b8fCQSRmgDBU"),
            temperature=0.3,
        )

        prompt = PromptTemplate.from_template(
            """You are an expert network security response agent. Your task is to analyze network threats 
            and generate appropriate mitigation strategies for SDN controllers.

            You have access to the following tools:
            {tools}

            Use these tools to analyze the threat and provide detailed mitigation strategies including:
            1. Specific SDN flow rules to block malicious traffic
            2. Latest research about effective mitigation techniques
            3. Best practices for network threat response
            4. Priority levels and implementation considerations

            Network Environment:
            - SDN Controller: Ryu at {ryu_host}
            - Topology: Linear topology with switches and hosts
            - Protocol Support: TCP, UDP, ICMP
            - Host Mapping: 10.0.0.1->h1, 10.0.0.2->h2, etc.

            Remember to:
            - Generate precise OpenFlow rules for the Ryu controller
            - Consider the network topology and potential impact
            - Search for latest threat mitigation research
            - Provide clear justification for mitigation choices
            - Include priority levels and execution order

            To use a tool, please use the following format:
            Thought: I need to research mitigation strategies for this threat
            Action: the action to take, should be one of [{tool_names}]
            Action Input: (provide search terms for research tools)
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know how to create effective mitigation
            Final Answer: the final mitigation strategy and commands

            Begin!

            Question: {input}

            {agent_scratchpad}"""
        )

        # Create React agent
        agent = create_react_agent(llm=llm, tools=self.tools, prompt=prompt)

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=300,  # 5 minutes timeout
        )

    def __initialize_tools(self) -> None:
        """
        Initialize tools for the response agent to use for intelligent mitigation generation.
        """

        def get_network_topology() -> Dict[str, Any]:
            """
            Retrieve current network topology from Ryu controller.
            Returns information about switches, hosts, and their connections.
            """
            return self.query_ryu_topology()

        def get_flow_table_stats() -> Dict[str, Any]:
            """
            Get current flow table statistics from all switches.
            Returns flow counts and existing rules for analysis.
            """
            try:
                topology_data = self.query_ryu_topology()
                stats = {}

                for dpid in topology_data["switches"]:
                    try:
                        flows_url = f"{self.ryu_base_url}/stats/flow/{dpid}"
                        response = requests.get(flows_url, timeout=5)
                        if response.status_code == 200:
                            flows = response.json()
                            stats[f"switch_{dpid}"] = {
                                "flow_count": len(flows.get(str(dpid), [])),
                                "flows": flows.get(str(dpid), []),
                            }
                    except Exception as e:
                        stats[f"switch_{dpid}"] = {"error": str(e)}

                return stats
            except Exception as e:
                return {"error": str(e)}

        def get_mitigation_patterns() -> List[str]:
            """
            Get common mitigation patterns and strategies used in SDN networks.
            Returns a list of proven mitigation approaches.
            """
            return [
                "Source IP blocking - Drop all traffic from malicious IP",
                "Protocol-specific blocking - Block specific protocols (TCP/UDP/ICMP)",
                "Port-based filtering - Block traffic to/from specific ports",
                "Rate limiting - Limit packet rate from suspicious sources",
                "Traffic redirection - Redirect suspicious traffic for analysis",
                "VLAN isolation - Isolate compromised hosts to separate VLAN",
                "Quality of Service (QoS) - Downgrade priority of suspicious traffic",
                "Geographic blocking - Block traffic from specific geographic regions",
            ]

        # Create tools for the agent
        self.tools = [
            Tool(
                name="GetNetworkTopology",
                func=lambda _: get_network_topology(),
                description=get_network_topology.__doc__,
            ),
            Tool(
                name="GetFlowTableStats",
                func=lambda _: get_flow_table_stats(),
                description=get_flow_table_stats.__doc__,
            ),
            Tool(
                name="GetMitigationPatterns",
                func=lambda _: get_mitigation_patterns(),
                description=get_mitigation_patterns.__doc__,
            ),
            safe_web_search_tool,
            safe_arxiv_retrieve_tool,
        ]

        # Only allow the long-term memory access tool during inference
        if self.use_long_term_memory:
            self.tools.append(
                Tool(
                    name="AccessPreviousMitigationStrategies",
                    func=lambda query: self.ltm_db.retrieve_relevant_knowledge(
                        query=query
                    ),
                    description="Retrieve previous successful mitigation strategies from long-term memory based on similar threats and network conditions.",
                )
            )

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
        Generate mitigation commands using RAG-enhanced intelligent analysis.
        Falls back to traditional rule-based approach if intelligent generation fails.
        """
        if classification_results.get("final_prediction", "Unknown") == "Benign":
            print("No mitigation needed for benign traffic")
            return []

        # Extract threat information
        threat_info = {
            "type": classification_results.get("final_prediction", "Unknown"),
            "confidence": classification_results.get("confidence", "Unknown"),
            "severity": (
                "High"
                if classification_results.get("final_prediction", "Unknown") != "Benign"
                else "Low"
            ),
            "source": "ML Classification",
        }

        try:
            # Use intelligent RAG-based mitigation generation
            intelligent_result = self.get_intelligent_mitigation(
                threat_info=threat_info,
                network_sample=original_sample,
                classification_results=classification_results,
            )

            if (
                intelligent_result.get("intelligent_analysis")
                and intelligent_result["mitigation_strategy"]["commands"]
            ):
                print("✅ Using intelligent RAG-based mitigation strategy")
                return intelligent_result["mitigation_strategy"]["commands"]
            else:
                print("⚠️ Intelligent generation failed, using fallback")
                return intelligent_result["mitigation_strategy"]["commands"]

        except Exception as e:
            print(f"❌ Error in intelligent mitigation: {str(e)}")
            # Traditional fallback approach
            return self._generate_traditional_mitigation_commands(
                classification_results, original_sample, dpid
            )

    def _generate_traditional_mitigation_commands(
        self,
        classification_results: Dict[str, Any],
        original_sample: pd.DataFrame,
        dpid: int = 1,
    ) -> List[MitigationCommand]:
        """
        Traditional rule-based mitigation command generation (fallback method).
        """
        print("Using traditional rule-based mitigation generation")

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

    def get_rag_enhanced_summary(
        self,
        commands: List[MitigationCommand],
        threat_info: Dict[str, Any] = None,
        include_research: bool = True,
    ) -> str:
        """
        Get RAG-enhanced summary with research-backed explanations.

        Args:
            commands: List of mitigation commands
            threat_info: Information about the threat
            include_research: Whether to include research findings

        Returns:
            Enhanced summary with research context
        """
        try:
            if not include_research or not hasattr(self, "agent_executor"):
                return self.get_mitigation_summary(commands)

            # Create research prompt
            threat_type = (
                threat_info.get("type", "Unknown") if threat_info else "Unknown"
            )
            prompt = f"""Provide a comprehensive analysis of these network security mitigation strategies for {threat_type} attacks:

MITIGATION COMMANDS:
{self._get_command_summary(commands)}

Please research and provide:
1. Effectiveness analysis of these mitigation approaches
2. Latest research findings on {threat_type} attack mitigation
3. Potential limitations or side effects
4. Best practices and recommendations
5. Alternative or complementary strategies

Focus on practical insights for network security operators."""

            # Get intelligent analysis
            response = self.agent_executor.invoke({"input": prompt})

            return f"""🔍 INTELLIGENT MITIGATION ANALYSIS

{response.get('output', 'Analysis unavailable')}

📋 GENERATED COMMANDS:
{self._get_command_summary(commands)}

Generated using RAG-enhanced threat analysis."""

        except Exception as e:
            print(f"Error generating RAG-enhanced summary: {str(e)}")
            return f"""⚠️ TRADITIONAL MITIGATION SUMMARY

{self._get_command_summary(commands)}

Note: Enhanced analysis unavailable due to: {str(e)}"""

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
            "rag_capabilities": {
                "long_term_memory_enabled": self.use_long_term_memory,
                "intelligent_analysis_available": hasattr(self, "agent_executor"),
                "knowledge_base_size": (
                    len(self.ltm_db.get_all_knowledge())
                    if hasattr(self.ltm_db, "get_all_knowledge")
                    else 0
                ),
                "tools_available": len(self.tools) if hasattr(self, "tools") else 0,
            },
        }

    def reset_blocking_state(self):
        """Reset all tracking of blocked hosts and commands (for testing)"""
        self.blocked_hosts.clear()
        self.blocked_flows.clear()
        self.applied_commands.clear()
        print("Reset all blocking state")

    def get_intelligent_mitigation(
        self,
        threat_info: Dict[str, Any],
        network_sample: pd.DataFrame = None,
        classification_results: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Use RAG-enhanced LLM to generate intelligent mitigation strategies.

        Args:
            threat_info: Information about the detected threat
            network_sample: Original network traffic sample
            classification_results: Results from threat classification

        Returns:
            Dictionary with intelligent mitigation strategy and commands
        """
        try:
            # Build comprehensive threat context
            threat_context = self._build_threat_context(
                threat_info, network_sample, classification_results
            )

            prompt = f"""Analyze this network security threat and generate comprehensive mitigation strategies:

THREAT ANALYSIS:
{threat_context}

REQUIREMENTS:
1. Generate specific OpenFlow rules for Ryu controller
2. Research latest mitigation techniques for this threat type
3. Consider network topology and minimize disruption
4. Provide implementation priority and rationale
5. Include monitoring and verification steps

Please provide:
- Detailed threat assessment
- Specific mitigation commands (curl format for Ryu)
- Implementation strategy and priorities
- Potential side effects and monitoring recommendations"""

            # Run the intelligent agent
            response = self.agent_executor.invoke({"input": prompt})

            # Parse the response to extract mitigation commands
            mitigation_strategy = self._parse_llm_mitigation_response(response)

            return {
                "threat_context": threat_context,
                "llm_response": response,
                "mitigation_strategy": mitigation_strategy,
                "intelligent_analysis": True,
            }

        except Exception as e:
            print(f"Error in intelligent mitigation generation: {str(e)}")
            # Fallback to traditional mitigation
            return self._fallback_mitigation(threat_info, network_sample)

    def _build_threat_context(
        self,
        threat_info: Dict[str, Any],
        network_sample: pd.DataFrame = None,
        classification_results: Dict[str, Any] = None,
    ) -> str:
        """Build comprehensive context about the threat for LLM analysis."""
        context_parts = []

        # Basic threat information
        if threat_info:
            context_parts.append(f"Threat Type: {threat_info.get('type', 'Unknown')}")
            context_parts.append(f"Severity: {threat_info.get('severity', 'Unknown')}")
            context_parts.append(f"Source: {threat_info.get('source', 'Unknown')}")

        # Classification results
        if classification_results:
            context_parts.append(
                f"Classification: {classification_results.get('final_prediction', 'Unknown')}"
            )
            context_parts.append(
                f"Confidence: {classification_results.get('confidence', 'Unknown')}"
            )

        # Network sample details
        if network_sample is not None and not network_sample.empty:
            sample_info = []
            for col in ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"]:
                if col in network_sample.columns:
                    value = (
                        network_sample[col].iloc[0]
                        if len(network_sample) > 0
                        else "N/A"
                    )
                    sample_info.append(f"{col}: {value}")
            context_parts.append("Network Sample: " + ", ".join(sample_info))

        # Current network state
        topology = self.query_ryu_topology()
        context_parts.append(f"Active Switches: {len(topology.get('switches', []))}")
        context_parts.append(f"Active Hosts: {len(topology.get('hosts', []))}")
        context_parts.append(f"Previously Blocked Hosts: {list(self.blocked_hosts)}")

        return "\n".join(context_parts)

    def _parse_llm_mitigation_response(
        self, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse LLM response to extract structured mitigation commands."""
        output = response.get("output", "")

        # Extract curl commands from the response
        import re

        curl_pattern = r"curl\s+[^`\n]+"
        curl_commands = re.findall(curl_pattern, output)

        # Convert to MitigationCommand objects
        commands = []
        for i, curl_cmd in enumerate(curl_commands):
            try:
                # Extract basic info from curl command
                command_type = "intelligent_mitigation"
                if "ipv4_src" in curl_cmd:
                    command_type = "block_host"
                elif "tcp_dst" in curl_cmd or "udp_dst" in curl_cmd:
                    command_type = "block_port"
                elif "ip_proto" in curl_cmd:
                    command_type = "block_protocol"

                cmd = MitigationCommand(
                    command_type=command_type,
                    description=f"Intelligent mitigation command {i+1} generated by RAG agent",
                    curl_command=curl_cmd.strip(),
                    priority=100 + i,  # Increase priority for each command
                    dpid=1,  # Default DPID
                )
                commands.append(cmd)
            except Exception as e:
                print(f"Error parsing command {curl_cmd}: {str(e)}")
                continue

        return {
            "commands": commands,
            "analysis": output,
            "command_count": len(commands),
            "strategy_type": "intelligent_rag",
        }

    def _fallback_mitigation(
        self, threat_info: Dict[str, Any], network_sample: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Fallback to traditional mitigation if intelligent generation fails."""
        print("Using fallback mitigation strategy")

        # Extract basic parameters for traditional mitigation
        src_ip = "10.0.0.1"  # Default
        if network_sample is not None and not network_sample.empty:
            if "Src IP" in network_sample.columns:
                src_ip = str(network_sample["Src IP"].iloc[0])

        # Use existing mitigation logic
        result = self.generate_and_run_mitigation_commands(src_ip=src_ip, execute=False)

        return {
            "threat_context": "Fallback mitigation used",
            "mitigation_strategy": {
                "commands": result["commands"],
                "analysis": "Traditional rule-based mitigation",
                "strategy_type": "fallback",
            },
            "intelligent_analysis": False,
        }

    def build_long_term_memory(
        self,
        threat_samples: List[Dict[str, Any]],
        successful_mitigations: List[Dict[str, Any]],
        rate_limit_threshold: int = 3,
    ) -> None:
        """
        Build long-term memory of successful mitigation strategies.

        Args:
            threat_samples: List of threat information and contexts
            successful_mitigations: List of successful mitigation responses
            rate_limit_threshold: Rate limiting for API calls
        """
        print("Building long-term memory for mitigation strategies...")

        # Store at most 25 successful mitigations
        for i in range(min(25, len(threat_samples))):
            attempts = 0

            # Initialize LLM for every sample to avoid rate limits
            self.__initialize_llm()

            threat_sample = threat_samples[i]
            successful_mitigation = (
                successful_mitigations[i] if i < len(successful_mitigations) else None
            )

            try:
                # Generate mitigation strategy
                response = self.get_intelligent_mitigation(threat_sample)

                # If we have a known successful mitigation, validate against it
                if successful_mitigation and response.get("intelligent_analysis"):
                    # Store successful strategy in long-term memory
                    self.ltm_db.add_knowledge(
                        f"""
                        Threat Context: {response['threat_context']}
                        
                        Successful Mitigation Strategy: {response['mitigation_strategy']['analysis']}
                        
                        Generated Commands: {[cmd.dict() for cmd in response['mitigation_strategy']['commands']]}
                        
                        Implementation Results: {successful_mitigation}
                        
                        LLM Response: {response['llm_response']}
                        """
                    )
                    print(
                        f"Added successful mitigation strategy {i+1} to long-term memory"
                    )

                attempts += 1

                # Rate limiting
                if attempts % rate_limit_threshold == 0:
                    time.sleep(30)

            except Exception as e:
                print(f"Error processing mitigation sample {i+1}: {str(e)}")
                continue
