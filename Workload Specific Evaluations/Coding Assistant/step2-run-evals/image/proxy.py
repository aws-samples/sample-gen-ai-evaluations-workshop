#!/usr/bin/env python3
"""Allowlist HTTP CONNECT proxy + relay. Only CONNECT host:443 for hosts in ALLOW; refuses non-public destinations."""
import asyncio, ipaddress, os, socket, sys, time

# NOTE: the real mantle host is bedrock-mantle.us-east-1.api.aws (the .amazonaws.com name does not resolve); keep both.
# The allowlist is the root-owned file /run/fw/allow.txt, re-read on every CONNECT. Empty or missing = deny all (fail closed).
# The environment rewrites it at each phase switch (the task's allowed_hosts of that phase).
ALLOW_FILE = "/run/fw/allow.txt"


def allowed():
    try:
        return {l.strip().lower() for l in open(ALLOW_FILE) if l.strip() and not l.startswith("#")}
    except OSError:
        return set()
# modes: `proxy.py serve SOCK`  allowlist proxy on a unix socket (main netns, user proxy)
#        `proxy.py relay PORT SOCK`  127.0.0.1:PORT -> SOCK, runs inside the agent's empty netns
MODE = sys.argv[1]
MAXCONN = 64
active = 0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


async def pipe(r, w):
    try:
        while (d := await r.read(65536)):
            w.write(d)
            await w.drain()
    except Exception:
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


async def handle(cr, cw):
    global active
    counted = False
    try:
        if active >= MAXCONN:
            raise PermissionError("too many")
        active += 1
        counted = True
        head = await asyncio.wait_for(cr.readuntil(b"\r\n\r\n"), 10)
        line = head.split(b"\r\n", 1)[0].decode("latin1")
        parts = line.split(" ")
        if len(parts) != 3 or parts[0] != "CONNECT":
            cw.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            log("DENY method", line[:80])
            return
        host, _, port = parts[1].rpartition(":")
        host = host.lower().rstrip(".")
        if port != "443" or host not in allowed():
            cw.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            log("DENY target", parts[1][:80])
            return
        infos = await asyncio.get_running_loop().getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
        ip = infos[0][4][0]
        if not ipaddress.ip_address(ip).is_global:
            cw.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            log("DENY non-global", host, ip)
            return
        ur, uw = await asyncio.wait_for(asyncio.open_connection(ip, 443), 10)
        cw.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await cw.drain()
        log("ALLOW", host, ip)
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
    except Exception as e:
        try:
            cw.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        except Exception:
            pass
        log("ERR", type(e).__name__, str(e)[:80])
    finally:
        if counted:
            active -= 1
        try:
            cw.close()
        except Exception:
            pass


async def relay(cr, cw):
    try:
        ur, uw = await asyncio.open_unix_connection(sys.argv[3])
    except Exception as e:
        log("relay ERR", e)
        cw.close()
        return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def main():
    if MODE == "serve":
        os.umask(0o077)
        srv = await asyncio.start_unix_server(handle, sys.argv[2])
    else:
        srv = await asyncio.start_server(relay, "127.0.0.1", int(sys.argv[2]))
    log("listening", MODE, sys.argv[2:])
    async with srv:
        await srv.serve_forever()


asyncio.run(main())
