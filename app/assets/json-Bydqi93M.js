import"./graph-C9AhnfEz.js";import{i as r}from"./_baseUniq-Bkki9EBs.js";import{c as e}from"./clone-CKE7aywN.js";import{m as n}from"./min-DFUqgUuu.js";function o(n){var o={options:{directed:n.isDirected(),multigraph:n.isMultigraph(),compound:n.isCompound()},nodes:t(n),edges:a(n)};return r(n.graph())||(o.value=e(n.graph())),o}function t(e){return n(e.nodes(),(function(n){var o=e.node(n),t=e.parent(n),a={v:n};return r(o)||(a.value=o),r(t)||(a.parent=t),a}))}function a(e){return n(e.edges(),(function(n){var o=e.edge(n),t={v:n.v,w:n.w};return r(n.name)||(t.name=n.name),r(o)||(t.value=o),t}))}export{o as w};
