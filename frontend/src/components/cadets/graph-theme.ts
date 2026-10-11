/* Read the shared UI tokens so the canvas and DOM use the same palette. */
export function graphStyles() {
  const tokens = getComputedStyle(document.documentElement);
  const color = (name: string) => tokens.getPropertyValue(name).trim();
  const text = color("--app-text");
  const canvas = color("--app-canvas");
  const danger = color("--app-danger");
  const success = color("--app-success");
  const selected = color("--graph-selected");
  const glyph = (node: any) => {
    const paths: Record<string, string> = {
      process: '<path d="m7 8 4 4-4 4m7 0h4"/>',
      file: '<path d="M14 3H6v18h12V7Zm0 0v5h4M9 12h6m-6 4h6"/>',
      host: '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/>',
      network: '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>',
    };
    const ink = node.data("alert") ? danger : success;
    return `data:image/svg+xml;utf8,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="${ink}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${paths[node.data("kind")] || paths.network}</svg>`)}`;
  };
  return [
    { selector: "node", style: {
      shape: "ellipse", width: 56, height: 56,
      "background-color": color("--app-success-soft"), "background-image": glyph,
      "background-width": 27, "background-height": 27,
      "border-width": 2, "border-color": success,
      label: "data(label)", color: text,
      "font-family": getComputedStyle(document.body).fontFamily,
      "font-size": 16, "font-weight": 600,
      "text-valign": "bottom", "text-margin-y": 11,
      "text-wrap": "wrap", "text-max-width": 160,
      "text-background-color": canvas, "text-background-opacity": .94,
      "text-background-padding": 3, "text-background-shape": "roundrectangle",
      "overlay-opacity": 0,
      "transition-property": "opacity, border-width, border-color", "transition-duration": "160ms",
    } },
    { selector: "node.threat", style: {
      width: 62, height: 62, "background-color": color("--app-danger-soft"),
      "border-color": danger, "border-width": 5, "border-style": "double",
    } },
    { selector: "node.selectedNode", style: { "outline-color": selected, "outline-width": 3, "outline-offset": 5, "outline-opacity": 1, "z-index": 999 } },
    { selector: "node.neighbor", style: { "outline-color": selected, "outline-width": 2, "outline-offset": 3, "outline-opacity": .65, "z-index": 500 } },
    { selector: "edge", style: {
      width: 1.5, "line-color": color("--graph-edge"), "target-arrow-color": color("--graph-edge"),
      "target-arrow-shape": "triangle", "curve-style": "bezier", "arrow-scale": .9,
      opacity: .75, label: "", "overlay-opacity": 0,
      "transition-property": "opacity, line-color, width, target-arrow-color", "transition-duration": "160ms",
    } },
    { selector: "edge.attackEdge", style: {
      width: 2.2, "line-color": danger, "target-arrow-color": danger, opacity: .9,
    } },
    { selector: "edge.benignEdge", style: { "line-color": success, "target-arrow-color": success, opacity: .65 } },
    { selector: "edge.mixedEdge", style: { "line-color": color("--graph-mixed"), "target-arrow-color": color("--graph-mixed"), opacity: .8 } },
    { selector: "edge.activeEdge", style: {
      width: 2.8, "line-color": selected, "target-arrow-color": selected, opacity: 1,
      label: "data(label)", color: text, "font-size": 12, "font-weight": 600,
      "font-family": getComputedStyle(document.body).fontFamily,
      "text-background-color": canvas, "text-background-opacity": .98,
      "text-background-padding": 4, "text-rotation": "autorotate", "z-index": 700,
    } },
    { selector: ".dim", style: { opacity: .72 } },
    { selector: ".filteredOut", style: { display: "none" } },
  ];
}
