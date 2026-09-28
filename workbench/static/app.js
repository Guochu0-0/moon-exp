// 路由：#/ 是画布，#/exp/<id> 是节点页。两个页面各自一个模块，共用 api.js。
// 画布只隐藏不销毁，从节点页返回时让该实验居中并选中。
import { mountCanvas } from './canvas.js';
import { mountNodePage } from './nodepage.js';

const canvas = mountCanvas(document.getElementById('canvas'), { onOpen: id => { location.hash = `#/exp/${encodeURIComponent(id)}`; } });
const page = mountNodePage(document.getElementById('page'), { onBack: id => { back = id; location.hash = '#/'; } });
let back = null;

function route() {
  const m = location.hash.match(/^#\/exp\/(.+)$/);
  if (m) {
    const id = decodeURIComponent(m[1]);
    canvas.hide(); page.show(id); back = id;
  } else {
    page.hide(); canvas.show(back); back = null;
  }
}
window.addEventListener('hashchange', route);
route();
