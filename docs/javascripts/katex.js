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
});
