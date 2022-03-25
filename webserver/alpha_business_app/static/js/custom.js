function uuidv4() {
	// https://stackoverflow.com/questions/105034/how-to-create-a-guid-uuid
	return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
		(c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
	);
}
  

$(document).ready(function() {
	$(".add-more").click(function(){
		var html = $("<div/>").html($(".copy").html());
		var target_replaces =  html.find('.target-replace');
		var uuid = uuidv4();
		$(target_replaces[0]).attr("data-bs-target", "#collapseAgent" + uuid);
		$(target_replaces[1]).attr("id", "collapseAgent" + uuid);
		$(".after-add-more").after(html);
	});
});
