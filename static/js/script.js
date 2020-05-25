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



	if (browserSupportFileUpload()) {
		document.getElementById('txtFileUpload').addEventListener('change', uploadFile, false);
	} else {
		$("#introHeader").html('The File APIs is not fully supported in this browser. Please use another browser.');
	}











	function JSON2CSV(objArray) {
	    var array = typeof objArray != 'object' ? JSON.parse(objArray) : objArray;
	    var str = '';
	    var line = '';

	    if ($("#labels").is(':checked')) {
	        var head = array[0];
	        if ($("#quote").is(':checked')) {
	            for (var index in array[0]) {
	                var value = index + "";
	                line += '"' + value.replace(/"/g, '""') + '",';
	            }
	        } else {
	            for (var index in array[0]) {
	                line += index + ',';
	            }
	        }

	        line = line.slice(0, -1);
	        str += line + '\r\n';
	    }

	    for (var i = 0; i < array.length; i++) {
	        var line = '';

	        if ($("#quote").is(':checked')) {
	            for (var index in array[i]) {
	                var value = array[i][index] + "";
	                line += '"' + value.replace(/"/g, '""') + '",';
	            }
	        } else {
	            for (var index in array[i]) {
	                line += array[i][index] + ',';
	            }
	        }

	        line = line.slice(0, -1);
	        str += line + '\r\n';
	    }
	    return str;
	}


	function download(text) {
	    // var element = document.createElement('a');
	    // element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
	    // element.setAttribute('download', 'cardio_predictions.txt');
	    // element.style.display = 'none';
	    // document.body.appendChild(element);
	    // element.click();
	    // document.body.removeChild(element);

	    var json = $.parseJSON(text);
		var csv = JSON2CSV(json);
		var downloadLink = document.createElement("a");
		var blob = new Blob(["\ufeff", csv]);
		var url = URL.createObjectURL(blob);
		downloadLink.href = url;
		downloadLink.download = "house_rent_predictions.csv";

		document.body.appendChild(downloadLink);
		downloadLink.click();
		document.body.removeChild(downloadLink);
	}

    var my_text = document.getElementById("result_val").innerText;	
    if(my_text != ""){
    	var res = confirm("Download final Prediction?\nThe file will be downloaded in txt format with house id and predicted rent price. \n\n{{any_message_multi}}");
    	if(res){
    		download(my_text);
    	}
    }				 