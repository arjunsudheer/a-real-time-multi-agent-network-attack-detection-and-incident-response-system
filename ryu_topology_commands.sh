#!/bin/bash
# Ryu Controller Topology Query Commands
# Usage: ./ryu_topology_commands.sh or run individual commands

RYU_HOST="localhost:8080"

echo "=== Ryu Controller Topology Information ==="
echo "Controller: http://$RYU_HOST"
echo

# 1. Get all switches
echo "1. All Switches:"
curl -s -X GET "http://$RYU_HOST/stats/switches" | python3 -m json.tool 2>/dev/null || curl -s -X GET "http://$RYU_HOST/stats/switches"
echo -e "\n"

# 2. Get topology switches (with detailed info)
echo "2. Topology Switches:"
curl -s -X GET "http://$RYU_HOST/v1.0/topology/switches" | python3 -m json.tool 2>/dev/null || curl -s -X GET "http://$RYU_HOST/v1.0/topology/switches"
echo -e "\n"

# 3. Get topology links
echo "3. Topology Links:"
curl -s -X GET "http://$RYU_HOST/v1.0/topology/links" | python3 -m json.tool 2>/dev/null || curl -s -X GET "http://$RYU_HOST/v1.0/topology/links"
echo -e "\n"

# 4. Get topology hosts
echo "4. Topology Hosts:"
curl -s -X GET "http://$RYU_HOST/v1.0/topology/hosts" | python3 -m json.tool 2>/dev/null || curl -s -X GET "http://$RYU_HOST/v1.0/topology/hosts"
echo -e "\n"

# 5. Get switch descriptions for each switch
echo "5. Switch Descriptions:"
SWITCHES=$(curl -s -X GET "http://$RYU_HOST/stats/switches" | grep -o '[0-9]\+' || echo "1 2 3")
for dpid in $SWITCHES; do
    echo "  Switch DPID $dpid:"
    curl -s -X GET "http://$RYU_HOST/stats/desc/$dpid" | python3 -m json.tool 2>/dev/null || curl -s -X GET "http://$RYU_HOST/stats/desc/$dpid"
    echo
done

# 6. Get ports for each switch
echo "6. Switch Ports:"
for dpid in $SWITCHES; do
    echo "  Switch DPID $dpid ports:"
    curl -s -X GET "http://$RYU_HOST/stats/port/$dpid" | python3 -m json.tool 2>/dev/null || curl -s -X GET "http://$RYU_HOST/stats/port/$dpid"
    echo
done

# 7. Get flow entries for each switch
echo "7. Flow Entries:"
for dpid in $SWITCHES; do
    echo "  Switch DPID $dpid flows:"
    curl -s -X GET "http://$RYU_HOST/stats/flow/$dpid" | python3 -m json.tool 2>/dev/null || curl -s -X GET "http://$RYU_HOST/stats/flow/$dpid"
    echo
done

echo "=== Individual Commands ==="
echo "Basic topology commands you can run individually:"
echo
echo "# Get all switches"
echo "curl -X GET http://$RYU_HOST/stats/switches"
echo
echo "# Get topology switches with details"
echo "curl -X GET http://$RYU_HOST/v1.0/topology/switches"
echo
echo "# Get links between switches"
echo "curl -X GET http://$RYU_HOST/v1.0/topology/links"
echo
echo "# Get connected hosts"
echo "curl -X GET http://$RYU_HOST/v1.0/topology/hosts"
echo
echo "# Get switch description (replace {dpid} with actual switch ID)"
echo "curl -X GET http://$RYU_HOST/stats/desc/{dpid}"
echo
echo "# Get switch ports"
echo "curl -X GET http://$RYU_HOST/stats/port/{dpid}"
echo
echo "# Get flow entries"
echo "curl -X GET http://$RYU_HOST/stats/flow/{dpid}"
echo
echo "# Get port statistics"
echo "curl -X GET http://$RYU_HOST/stats/portdesc/{dpid}"
echo
echo "# Get aggregate flow statistics"
echo "curl -X GET http://$RYU_HOST/stats/aggregateflow/{dpid}"
echo
echo "# Get table statistics"
echo "curl -X GET http://$RYU_HOST/stats/table/{dpid}"
echo
echo "=== Examples with current switches ==="
for dpid in $SWITCHES; do
    echo "# Switch DPID $dpid commands:"
    echo "curl -X GET http://$RYU_HOST/stats/desc/$dpid"
    echo "curl -X GET http://$RYU_HOST/stats/port/$dpid"
    echo "curl -X GET http://$RYU_HOST/stats/flow/$dpid"
    echo
done 