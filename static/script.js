function loginDoctor() {
  const username = document.getElementById("loginUsername").value;
  const password = document.getElementById("loginPassword").value;

  const savedUsername = localStorage.getItem("doctorUsername");
  const savedPassword = localStorage.getItem("doctorPassword");

  if (username === savedUsername && password === savedPassword) {
   
    window.location.href = "index.html";
    return false; 
  } else {
    alert("Invalid username or password");
    return false; 
  }
}

function signupDoctor() {
  const username = document.getElementById("signupUsername").value;
  const email = document.getElementById("signupEmail").value;
  const password = document.getElementById("signupPassword").value;

  
  localStorage.setItem("doctorUsername", username);
  localStorage.setItem("doctorPassword", password);
  localStorage.setItem("doctorEmail", email);

  alert("Signup successful! You can now log in.");
  window.location.href = "login.html";
  return false; 
}
