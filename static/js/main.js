// Get the border box element
var borderBox = document.getElementById("uploaded-file-box");

// Add an event listener to the "Predict" button
var predictButton = document.querySelector("input[type='submit']");
predictButton.addEventListener("click", function() {

  // Display the border box
  borderBox.classList.remove("hidden");

});