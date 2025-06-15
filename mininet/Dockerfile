FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y mininet openvswitch-switch hping3 nmap sudo iproute2 && \
    rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash", "-c", "sudo mn --controller=remote,ip=$ryu_ip --topo=linear,3 --mac"]