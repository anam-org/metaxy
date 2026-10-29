/**
 * Clickable glossary tooltips for Material for MkDocs
 *
 * Glossary format: *[term]: Description | /path/to/docs
 * The URL after the pipe becomes a clickable "Learn more" link in the tooltip.
 */
(function () {
  let tooltip = null;
  let hideTimeout = null;
  // Cached once on first init() when __config is guaranteed fresh.
  // Material's instant navigation doesn't update the DOM's __config script,
  // so reading it after SPA navigation gives a stale base value.
  let siteRoot = null;

  function createTooltip() {
    tooltip = document.createElement("div");
    tooltip.className = "glossary-tooltip";
    tooltip.style.cssText = `
      position: fixed;
      z-index: 1000;
      background: var(--md-default-bg-color, #fff);
      color: var(--md-default-fg-color, #333);
      border: 1px solid var(--md-default-fg-color--lightest, #ddd);
      border-radius: 4px;
      padding: 0.6em 0.8em;
      max-width: 400px;
      font-size: 0.8rem;
      line-height: 1.5;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      pointer-events: auto;
      display: none;
    `;
    document.body.appendChild(tooltip);

    tooltip.addEventListener("mouseenter", function () {
      clearTimeout(hideTimeout);
    });

    tooltip.addEventListener("mouseleave", function () {
      hideTooltip();
    });
  }

  function showTooltip(abbr, description, url) {
    clearTimeout(hideTimeout);

    let html = '<span class="glossary-description">' + escapeHtml(description) + "</span>";
    if (url) {
      html +=
        '<a href="' +
        escapeHtml(resolveUrl(url)) +
        '" target="_blank" class="glossary-link" style="display: block; margin-top: 0.4em; color: var(--md-accent-fg-color, #526cfe); text-decoration: none; font-weight: 500;">Learn more →</a>';
    }
    tooltip.innerHTML = html;

    const link = tooltip.querySelector(".glossary-link");
    if (link) {
      link.addEventListener("mouseenter", function () {
        link.style.textDecoration = "underline";
      });
      link.addEventListener("mouseleave", function () {
        link.style.textDecoration = "none";
      });
    }

    const rect = abbr.getBoundingClientRect();
    tooltip.style.display = "block";

    // Position below the term
    let top = rect.bottom + 8;
    let left = rect.left;

    // Adjust if tooltip goes off-screen
    const tooltipRect = tooltip.getBoundingClientRect();
    if (left + tooltipRect.width > window.innerWidth - 16) {
      left = window.innerWidth - tooltipRect.width - 16;
    }
    if (top + tooltipRect.height > window.innerHeight - 16) {
      top = rect.top - tooltipRect.height - 8;
    }

    tooltip.style.top = top + "px";
    tooltip.style.left = Math.max(8, left) + "px";
  }

  function hideTooltip() {
    hideTimeout = setTimeout(function () {
      if (tooltip) {
        tooltip.style.display = "none";
      }
    }, 100);
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function resolveUrl(url) {
    if (!url.startsWith("/")) return url;
    if (!siteRoot) return url;
    return new URL(url.slice(1), siteRoot).href;
  }

  function init() {
    if (!tooltip) {
      createTooltip();
    }

    if (siteRoot === null) {
      var configEl = document.getElementById("__config");
      if (configEl) {
        try {
          var root = new URL(
            JSON.parse(configEl.textContent).base,
            location,
          ).href;
          siteRoot = root.endsWith("/") ? root : root + "/";
        } catch (e) {
          siteRoot = "";
        }
      }
    }

    // Process all abbreviations and disable Material's tooltip
    document.querySelectorAll("abbr[title]").forEach(function (abbr) {
      const title = abbr.getAttribute("title");
      const pipeIndex = title.lastIndexOf(" | ");

      let description, url;
      if (pipeIndex !== -1) {
        description = title.substring(0, pipeIndex);
        url = title.substring(pipeIndex + 3);
      } else {
        description = title;
        url = null;
      }

      // Store data and remove title to disable browser/Material tooltip
      abbr.setAttribute("data-glossary-desc", description);
      if (url) {
        abbr.setAttribute("data-glossary-url", url);
      }
      abbr.removeAttribute("title");

      // Remove default abbr styling
      abbr.style.textDecoration = "none";
      abbr.style.cursor = "help";

      abbr.addEventListener("mouseenter", function () {
        const desc = abbr.getAttribute("data-glossary-desc");
        const link = abbr.getAttribute("data-glossary-url");
        showTooltip(abbr, desc, link);
      });

      abbr.addEventListener("mouseleave", function () {
        hideTooltip();
      });
    });
  }

  // Initial load
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Support Material for MkDocs instant navigation
  if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
      init();
    });
  } else {
    // Fallback: re-init on navigation events
    document.addEventListener("DOMContentLoaded", function () {
      if (typeof document$ !== "undefined") {
        document$.subscribe(function () {
          init();
        });
      }
    });
  }
})();
