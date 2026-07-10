from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length=int(self.headers.get('Content-Length',0))
        body=json.loads(self.rfile.read(length).decode('utf-8') or '{}')
        prompt=''
        for m in body.get('messages',[]):
            if m.get('role')=='user': prompt=m.get('content','')
        response={
            'id':'mock','object':'chat.completion','model':body.get('model','mock-model'),
            'choices':[{'index':0,'message':{'role':'assistant','content':'Mock answer: pipeline works.'},'finish_reason':'stop'}]
        }
        data=json.dumps(response).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers(); self.wfile.write(data)

if __name__=='__main__':
    HTTPServer(('0.0.0.0',8000), Handler).serve_forever()
