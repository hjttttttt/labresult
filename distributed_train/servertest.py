import xmlrpc.server

class MyServer:
    def add(self, a, b):
        return a + b

server = xmlrpc.server.SimpleXMLRPCServer(("0.0.0.0", 8000))
server.register_instance(MyServer())
print("Server running...")
server.serve_forever()