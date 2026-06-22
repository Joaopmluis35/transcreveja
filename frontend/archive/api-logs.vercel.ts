/**
 * Legado Vercel — não usado no Cloudflare Pages.
 * Os logs do backoffice vêm de api.ouviescrevi.pt (FastAPI).
 */
import type { VercelRequest, VercelResponse } from '@vercel/node';
import fs from 'fs';
import path from 'path';

export default function handler(req: VercelRequest, res: VercelResponse) {
  const logPath = path.join('/tmp', 'ouviescrevi_frontend.log');

  if (req.method === 'POST') {
    const { message } = req.body || {};
    fs.appendFileSync(logPath, `${new Date().toISOString()} ${message}\n`);
    return res.status(200).json({ ok: true });
  }

  if (req.method === 'GET') {
    if (!fs.existsSync(logPath)) {
      return res.status(200).json({ logs: [] });
    }
    const data = fs.readFileSync(logPath, 'utf-8').split('\n').filter(Boolean);
    return res.status(200).json({ logs: data });
  }

  res.status(405).json({ error: 'Method not allowed' });
}
