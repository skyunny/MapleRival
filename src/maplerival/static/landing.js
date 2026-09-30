const form = document.querySelector("#characterSearch");
const input = document.querySelector("#characterName");
const message = document.querySelector("#searchMessage");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = input.value.trim();
  if (!name) return;
  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "···";
  message.className = "message";
  message.textContent = `${name}님의 데이터를 불러오고 있습니다. 잠시만 기다려 주세요.`;
  try {
    const response = await fetch("/api/characters/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "캐릭터를 찾지 못했습니다.");
    window.location.href = `/dashboard?owner=${encodeURIComponent(result.name)}`;
  } catch (error) {
    message.className = "message error";
    message.textContent = error.message;
    button.disabled = false;
    button.textContent = "→";
  }
});
