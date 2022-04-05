$(document).ready(function() {
	$('button.add-more').bind("click", function () {
		var self = $(this)
		$.ajax({url: self.data("url"),
		success: function (data) {
			self.before(data)
		}
		});
	});
});
