$(document).ready(function() {
	$("button.add-more").click("click", function () {
		// adds the return value of the ajax call (html) before the element.
		var self = $(this)
		$.ajax({url: self.data("url"),
		success: function (data) {
			self.before(data)
		}
		});
	});

	$("button.replace-me").click("click", function () {
		var self = $(this)
		$.ajax({url: self.data("url"),
		success: function (data) {
			self.replaceWith(data)
			console.log(data)
		}
		});
	});

	$("select.task-selection").change(function () {
		// displays the monitoring options when 'monitoring' is selected
		if(this.value == "monitoring") {
			$('.hide-not-monitoring').each(function () {
				$(this).removeClass("d-none")
			});			
		} else {
			$('.hide-not-monitoring').each(function () {
				$(this).addClass("d-none")
			});
		}
	}).trigger('change');


});
