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
		// replaces the element by the element returned by ajax (html)
		var statusButton = $("button.replace-me")
		$.ajax({url: statusButton.data("url"),
		success: function (data) {
			statusButton.replaceWith(data);
		}
		});
	};

	// check for API status all 5 seconds
	window.setInterval(function() {updateAPIHealth()}, 5000)

	$("select.task-selection").change(function () {
		// displays the monitoring options when 'monitoring' is selected
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
});
