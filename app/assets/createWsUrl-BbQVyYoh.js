import{S as s,K as t}from"./index-UhR0gcZS.js";function n(s){const n=new URLSearchParams(window.location.search);return n.set(t.sessionId,s),o(`ws?${n.toString()}`)}function o(t){if(t.startsWith("ws:")||t.startsWith("wss:"))return t;const n=new URL(document.baseURI),o="https:"===n.protocol?"wss:":"ws:",a=n.host,r=n.pathname;return`${o}//${a}${s.withoutTrailingSlash(r)}/${s.withoutLeadingSlash(t)}`}export{n as c,o as r};
