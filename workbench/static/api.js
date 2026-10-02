// 画布与节点页共用的 API 客户端。接口说明见 workbench/server.py 顶部。

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function call(method, path, body) {
  const init = { method, headers: {} };
  if (body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
  const r = await fetch(path, init);
  const d = await r.json().catch(() => ({ error: `${r.status} ${r.statusText}` }));
  if (!r.ok) throw new ApiError(d.error || `${r.status} ${r.statusText}`, r.status);
  return d;
}

const exp = id => `/api/exp/${encodeURIComponent(id)}`;

export const api = {
  data: () => call('GET', '/api/data'),
  detail: id => call('GET', exp(id)),
  compare: (method, ref, split) => call('GET', `/api/compare?${new URLSearchParams({ method, split, ...(ref ? { ref } : {}) })}`),
  saveNotes: (id, text) => call('PUT', `${exp(id)}/notes`, { text }),
  create: body => call('POST', '/api/exp', body),
  setParent: (id, parent, init = null) => call('POST', `${exp(id)}/parent`, { parent, init }),
  setTitle: (id, title) => call('POST', `${exp(id)}/title`, { title }),
  rename: (id, newId) => call('POST', `${exp(id)}/rename`, { id: newId }),
  remove: id => call('DELETE', exp(id)),
  saveCanvas: body => call('PUT', '/api/canvas', body),
};
