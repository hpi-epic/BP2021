var ws = new WebSocket("ws://192.168.159.134:8000/ws");
ws.onmessage = function(event) {
	var messages = document.getElementById('container-crashed-alert')
	var message = document.createElement('div')
	var content = document.createTextNode(event.data)
	message.appendChild(content)
	messages.appendChild(message)
};
