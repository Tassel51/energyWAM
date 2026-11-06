"""Loopback-only CONNECT relay restricted to official Box download hosts."""
import socket,socketserver,select,threading,time
ALLOWED={'utexas.box.com','utexas.app.box.com','public.boxcloud.com','app.box.com','dl.boxcloud.com'}
class Handler(socketserver.BaseRequestHandler):
 def handle(self):
  upstream=None
  try:
   self.request.settimeout(15);header=b''
   while not header.endswith(b'\r\n\r\n'):
    b=self.request.recv(1)
    if not b or len(header)>16384:return
    header+=b
   first=header.split(b'\r\n')[0].decode().split()
   if len(first)!=3 or first[0]!='CONNECT':return
   host,port=first[1].rsplit(':',1)
   if host not in ALLOWED or port!='443':
    self.request.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n');return
   upstream=socket.create_connection(('127.0.0.1',27904),timeout=15)
   upstream.sendall(('CONNECT '+host+':443 HTTP/1.1\r\nHost: '+host+':443\r\n\r\n').encode())
   response=b''
   while not response.endswith(b'\r\n\r\n'):
    b=upstream.recv(1)
    if not b or len(response)>16384:return
    response+=b
   if b' 200 ' not in response.split(b'\r\n')[0]:return
   self.request.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
   self.request.settimeout(None);upstream.settimeout(None)
   peers={self.request:upstream,upstream:self.request}
   while peers:
    ready,_,_=select.select(list(peers),[],[],60)
    for source in ready:
     data=source.recv(256*1024)
     if data:peers[source].sendall(data)
     else:
      try:peers[source].shutdown(socket.SHUT_WR)
      except OSError:pass
      del peers[source]
  except OSError:pass
  finally:
   if upstream:upstream.close()
class Server(socketserver.ThreadingTCPServer):
 allow_reuse_address=True;daemon_threads=True
 def handle_error(self,*args):pass
if __name__=='__main__':
 with Server(('127.0.0.1',38740),Handler) as server:
  print('Box-only loopback relay ready on 38740',flush=True);server.serve_forever()
