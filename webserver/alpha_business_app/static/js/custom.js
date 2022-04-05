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

	function replace(element) {
		// replaces the element by the element returned by ajax (html) and adds this click event to it
		$.ajax({url: element.data("url"),
		success: function (data) {
			element.replaceWith(data);
			$("button.replace-me").click(function() {replace($(this))});
		}
		});
	};

	$("button.replace-me").click(function() {replace($(this))});


	$("select.task-selection").change(function () {
		// displays the monitoring options when 'monitoring' is selected
		if(this.value == "monitoring") {
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
