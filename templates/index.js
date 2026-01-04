async function analyze() {
  const message = document.getElementById("text").value;

  const response = await fetch("http://127.0.0.1:8000/api/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  const data = await response.json();
  const result = document.getElementById("result");
  result.innerHTML = JSON.stringify(data, null, 2);
}