// Runs before first paint (CSP forbids inline scripts): applies the saved theme so light mode doesn't flash dark.
try {
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
} catch (e) { /* storage unavailable: keep the dark default */ }
