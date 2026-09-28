// 节点页（#/exp/<id>）。完整内容由「节点页：实验信息、比较设置、结果与 Notes」那张票实现，这里先放占位页，
// 保证画布与节点页之间能往返。
import { api } from './api.js';

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function mountNodePage(root, { onBack }) {
  root.className = 'np';
  return {
    async show(id) {
      root.hidden = false;
      root.innerHTML = `<a class="back" href="#/">← 画布</a><h1><span class="nid">${esc(id)}</span></h1><p class="wip">节点页尚未实现。</p>`;
      root.querySelector('.back').addEventListener('click', ev => { ev.preventDefault(); onBack(id); });
      try {
        const e = (await api.data()).experiments.find(x => x.id === id);
        root.querySelector('h1').insertAdjacentHTML('beforeend', e ? ` ${esc(e.title)}` : '');
        if (!e) root.querySelector('.wip').textContent = `没有实验 ${id}。`;
      } catch (err) { root.querySelector('.wip').textContent = `读取数据失败：${err.message}`; }
    },
    hide() { root.hidden = true; root.innerHTML = ''; },
  };
}
