$('.message a').click(function(){
   $('form').animate({height: "toggle", opacity: "toggle"}, "slow");
});


	var browserSupportFileUpload = function() {
		var isCompatible = false;
		if(window.File && window.FileReader && window.FileList && window.Blob) {
			isCompatible = true;
		}
		return isCompatible;
	};

	// Upload selected file and create array
	var uploadFile = function(evt) {
		var file = evt.target.files[0];
		Papa.parse(file, {
			complete: function(results) {
				console.log("AAA: ", results);

				var data_arr = results.data;
				var element = data_arr.pop();
				console.log("remove last element: ", element);
				var myJSON = JSON.stringify(data_arr);
				console.log("PPPPPP: ", myJSON);


				var myForm = document.getElementById('upload-form')
				var hiddenInput = document.createElement('input')

				hiddenInput.type = 'hidden'
				hiddenInput.name = 'myarray'
				hiddenInput.value = myJSON

				myForm.appendChild(hiddenInput)
			}
		});
	};

function download_data(filename, text) {
    var element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);

    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}


window.addEventListener('online', () => console.log('came online'));
window.addEventListener('offline', () => console.log('came offline'));


if (browserSupportFileUpload()) {
		document.getElementById('txtFileUpload').addEventListener('change', uploadFile, false);
	} else {
		$("#introHeader").html('The File APIs is not fully supported in this browser. Please use another browser.');
	}
