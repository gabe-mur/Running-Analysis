// Entry point: wire the router and the upload handlers, then render.

import { route } from "./router.js";
import { registerUploads } from "./uploads.js";

registerUploads();
window.addEventListener("hashchange", route);
route();
