#!/bin/bash
service openvswitch-switch start
exec "$@" 