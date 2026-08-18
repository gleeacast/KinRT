function setupNavigation() {
  const toggle = document.querySelector("[data-nav-toggle]");
  const nav = document.querySelector("[data-primary-nav]");
  if (!toggle || !nav) return;

  toggle.addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", String(open));
  });

  nav.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      nav.classList.remove("is-open");
      toggle.setAttribute("aria-expanded", "false");
    }
  });
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}

function setupCopyButtons() {
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const block = button.closest(".code-block");
      const code = block?.querySelector("code");
      if (!code) return;
      const original = button.textContent;
      try {
        await copyText(code.textContent);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Copy failed";
      }
      window.setTimeout(() => { button.textContent = original; }, 1400);
    });
  });
}

function setupConfigFilter() {
  const input = document.querySelector("[data-config-filter]");
  const rows = [...document.querySelectorAll("[data-config-row]")];
  const count = document.querySelector("[data-config-count]");
  if (!input || !rows.length) return;

  const apply = () => {
    const query = input.value.trim().toLowerCase();
    let visible = 0;
    rows.forEach((row) => {
      const match = !query || row.textContent.toLowerCase().includes(query);
      row.hidden = !match;
      if (match) visible += 1;
    });
    if (count) count.textContent = `${visible} of ${rows.length} rows`;
  };
  input.addEventListener("input", apply);
  apply();
}

function setCurrentNavigation() {
  const page = document.body.dataset.page;
  if (!page) return;
  document.querySelectorAll(`[data-page-link="${page}"]`).forEach((link) => {
    link.setAttribute("aria-current", "page");
  });
}

setupNavigation();
setupCopyButtons();
setupConfigFilter();
setCurrentNavigation();
