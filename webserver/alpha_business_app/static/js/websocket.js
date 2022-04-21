var ws = new WebSocket("ws://192.168.159.134:8001/ws");
ws.onopen = function (event) {
	console.log('connection geöffnet');
}
ws.onmessage = function(event) {
	// var messages = document.getElementById('container-crashed-alert')
	// var message = document.createElement('div')
	// var content = document.createTextNode(event.data)
	// message.appendChild(content)
	// messages.appendChild(message)
	console.log(event.data);
}

ws.onclose = function(event) {
	console.log('zu')
}
