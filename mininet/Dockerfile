FROM ubuntu:22.04

RUN apt-get update && apt-get install -y mininet openvswitch-switch hping3 nmap

ENTRYPOINT ["mn", "--controller=remote,ip=172.17.0.3", "--topo=linear,3", "--mac"]