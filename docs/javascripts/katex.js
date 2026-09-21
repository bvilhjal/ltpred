/* Render TeX after Material injects the page. Libraries load first; this file
 * only registers the callback. throwOnError=false so a bad delimiter does not
 * blank the rest of the page. */
document$.subscribe(function () {
  if (typeof renderMathInElement !== "function") return;
  renderMathInElement(document.body, {
    delimiters: [
      {left: "$$", right: "$$", display: true},
      {left: "\\[", right: "\\]", display: true},
      {left: "$", right: "$", display: false},
      {left: "\\(", right: "\\)", display: false},
    ],
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  });

  /* auto-render wraps each display equation in an inline <span> around the
   * block-level .katex-display. An inline box containing a block box inflates
   * its line box to ~104px for a 24px equation, which shows up as ~58px of dead
   * vertical space above and below every displayed equation. Making the
   * wrapper a block collapses it to the equation's own height (gaps fall to
   * the intended 1em margins). Measured in headless Chrome, 2026-09-21. */
  document.querySelectorAll(".katex-display").forEach(function (display) {
    var wrapper = display.parentElement;
    if (wrapper && wrapper.tagName === "SPAN") wrapper.style.display = "block";
  });
});
