$(document).ready(function() {
	function addEventToAddMoreButton () {
		$("button.add-more").click(function () {
			// adds the return value of the ajax call (html) before the element.
			var self = $(this)
			$.ajax({url: self.data("url"),
			success: function (data) {
				self.before(data)
			}
			});
		});
	};
	addEventToAddMoreButton()

	function getFormData () {
		var form = $("form.config-form");
		var formdata = form.serializeArray();
		return formdata;
	};
	
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

	$("select.marketplace-selection").change(function () {
		// will be called when another marketplace has been selected
		var self = $(this);
		const csrftoken = getCookie("csrftoken");
		var all_agents = $("div.all-agents");
		$.ajax({
			type: "POST",
			url: self.data("url"),
			data: {
				csrfmiddlewaretoken: csrftoken,
				"marketplace": self.val(),
				"agents_html": all_agents.html()
			},
			success: function (data) {
				all_agents.empty().append(data);
				addEventToAddMoreButton();
				addChangeToAgent();
			}
		});
	}).trigger("change");

	function addChangeToAgent () {
		$("select.agent-agent-class").change(function () {
			// will be called when agent dropdown has changed, we need to change rl hyperparameter for that
			var self = $(this);
			const csrftoken = getCookie("csrftoken");
			$.ajax({
				type: "POST",
				url: self.data("url"),
				data: {
					csrfmiddlewaretoken: csrftoken,
					"agent": self.val()
				},
				success: function (data) {
					// our view returns the data in the form:
					// <agent specific parameters ; seperated>|<all possible parameters ; seperated>
					// The parameters are already real IDs in our template, we just need to split it up again.
					parameter_parts = data.split('|');
					all_parameters = parameter_parts[1].split(';');
					specific_parameters = parameter_parts[0].split(';');
					$(all_parameters).each(function (_ , value) {
						// show all possible rl parameters
						$("#" + value).removeClass('d-none')
					});
					$(specific_parameters).each(function (_ , value) {
						// make the specific ones not visible
						$("#" + value).addClass('d-none')
					});
				}
			});
		}).trigger("change");
	}
	addChangeToAgent()

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
		var self = $(this);
		var formdata = getFormData();

		const csrftoken = getCookie('csrftoken');
		$.ajax({
			type: "POST",
			url: self.data("url"),
			data: {
				csrfmiddlewaretoken: csrftoken,
				formdata
			},
			success: function (data) {
				$("p.notice-field").replaceWith(data);
			}
		});
	});
});
