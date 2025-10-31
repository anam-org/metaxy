(function () {
  var token = "22cb75dc-201e-4a72-9fb2-c3a53ce9207e";

  function addScarfPixel() {
    if (!document.body) {
      return;
    }

    var path = window.location.pathname || "index.html";
    var pageIdentifier = path === "/" ? "index.html" : path.replace(/^\//, "");

    var pixel = document.createElement("img");
    pixel.setAttribute("referrerpolicy", "no-referrer");
    pixel.src =
      "https://static.scarf.sh/a.png?x-pxid=" +
      token +
      "=" +
      encodeURIComponent(pageIdentifier);
    pixel.alt = "";
    pixel.width = 1;
    pixel.height = 1;
    pixel.style.position = "absolute";
    pixel.style.width = "1px";
    pixel.style.height = "1px";
    pixel.style.opacity = "0";
    pixel.style.pointerEvents = "none";
    pixel.style.border = "0";
    pixel.style.margin = "0";
    pixel.style.padding = "0";
    pixel.style.clip = "rect(0, 0, 0, 0)";

    document.body.appendChild(pixel);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addScarfPixel);
  } else {
    addScarfPixel();
  }
})();
