$(document).ready(function() {
	$("button.add-more").click(function () {
		// adds the return value of the ajax call (html) before the element.
		var self = $(this)
		$.ajax({url: self.data("url"),
		success: function (data) {
			self.before(data)
		}
		});
	});
	
	function updateAPIHealth() {
		// replaces the element by the element returned by ajax (html) and adds this click event to it
		var statusButton = $("button.replace-me")
		$.ajax({url: statusButton.data("url"),
		success: function (data) {
			statusButton.replaceWith(data);
			$("button.replace-me").click(function() {updateAPIHealth()})
		}
		});
	};

	updateAPIHealth();

	$("select.task-selection").change(function () {
		// displays the monitoring options when 'agent_monitoring' is selected
		var self = this
		if(self.value == "agent_monitoring") {
			$(".hide-not-monitoring").each(function () {
				$(this).removeClass("d-none")
			});			
		} else {
			$(".hide-not-monitoring").each(function () {
				$(this).addClass("d-none")
			});
		}
	}).trigger('change');

	function getCookie(name) {
		let cookieValue = null;
		if (document.cookie && document.cookie !== '') {
			const cookies = document.cookie.split(';');
			for (let i = 0; i < cookies.length; i++) {
				const cookie = cookies[i].trim();
				// Does this cookie string begin with the name we want?
				if (cookie.substring(0, name.length + 1) === (name + '=')) {
					cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
					break;
				}
			}
		}
		return cookieValue;
	}
	
	$("button.form-check").click(function () {
		$("table.config-status-display").remove();
		
		var self = $(this);
		var form = $("form.config-form");
		var formdata = form.serializeArray();

		const csrftoken = getCookie('csrftoken');
		$.ajax({
			type: "POST",
			url: self.data("url"),
			data: {
				csrfmiddlewaretoken: csrftoken,
				formdata
			},
			success: function (data) {
				const notice_field = $('#notice-field');
				if (notice_field.length > 0) {
					notice_field.replaceWith(data);
				} else {
					const endOfContent = document.getElementById('end-of-content');
					endOfContent.insertAdjacentHTML('afterend', data);
				}
			}
		});
	});

	var url = "ws://192.168.159.134:8001/ws";
	var ws = new WebSocket(url);
	ws.onopen = function (_) {
		console.log("connection to ", url, "open");
	}
	ws.onmessage = function(event) {
		const csrftoken = getCookie('csrftoken');
		$.ajax({
			type: "POST",
			url: "/container_notification",
			data: {
				csrfmiddlewaretoken: csrftoken,
				api_response: event.data
			},
			success: function (data) {
				var nav_bar = document.getElementById('main-nav-bar');
				nav_bar.insertAdjacentHTML('afterend', data)
			}
		});
		console.log(event.data);
	}

	ws.onclose = function(_) {
		console.log("connection to ", url, "closed");
	}
});
