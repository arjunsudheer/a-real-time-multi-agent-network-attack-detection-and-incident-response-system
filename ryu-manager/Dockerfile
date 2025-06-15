FROM python:3.9.18-slim
WORKDIR /app
RUN pip install ryu eventlet==0.30.2
EXPOSE 8080
ENTRYPOINT ["ryu-manager", "ryu.app.simple_switch_13", "ryu.app.ofctl_rest"] 