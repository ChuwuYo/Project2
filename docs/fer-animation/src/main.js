import { SCENES } from "./scenes/index.js";
import { initPlayer } from "./core/player.js";
import { initMobile } from "./core/mobile.js";

const player = initPlayer(SCENES);
initMobile(player);
