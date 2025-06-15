import json
import subprocess
import shlex
from typing import List, Dict, Any
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


class ResponseAgent:
    def __init__(self):
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
            print(f"\nGenerating mitigation commands:")
            print(
                f"Source IP: {src_ip} ({self.ip_to_host.get(src_ip, 'unknown host')})"
            )
            print(f"Destination IP: {dst_ip}")
            print(f"Protocol: {protocol}")
            print(f"Port: {dst_port}")

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
                print(f"\nExecuting: {cmd.description}")
                print(f"Command: {cmd.curl_command}")

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
                    print(f"✓ Successfully executed: {cmd.description}")
                else:
                    results["failed_commands"].append(
                        {
                            "command": cmd.description,
                            "curl_command": cmd.curl_command,
                            "error": result.stderr,
                            "status": "failed",
                        }
                    )
                    print(f"✗ Failed to execute: {cmd.description}")
                    print(f"Error: {result.stderr}")

            except Exception as e:
                results["failed_commands"].append(
                    {
                        "command": cmd.description,
                        "curl_command": cmd.curl_command,
                        "error": str(e),
                        "status": "error",
                    }
                )
                print(f"✗ Exception executing {cmd.description}: {str(e)}")

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
        return self._execute_commands(commands)

    def get_mitigation_summary(self, commands: List[MitigationCommand]) -> str:
        """Get summary of mitigation commands"""
        return self._get_command_summary(commands)
