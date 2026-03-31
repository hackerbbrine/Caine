'use strict';

const { app, BrowserWindow, Menu, shell, ipcMain } = require('electron');
const path = require('path');

const isDev = process.env.NODE_ENV === 'development';

// ── Security: disable remote module, keep Node out of renderer ──────────────
app.commandLine.appendSwitch('disable-features', 'OutOfBlinkCors');

function createWindow() {
  const win = new BrowserWindow({
    width:           1600,
    height:          950,
    minWidth:        1200,
    minHeight:       700,
    backgroundColor: '#080810',
    title:           'CAINE — Mission Control',
    icon:            path.join(__dirname, 'assets', 'icon.png'),   // optional
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      nodeIntegration:  false,
      contextIsolation: true,
      webSecurity:      true,
      // Allow loading Three.js from jsDelivr CDN
      // CSP is also set via meta tag in index.html
    },
    show: false,   // show after ready-to-show to avoid white flash
  });

  win.loadFile(path.join(__dirname, 'index.html'));

  win.once('ready-to-show', () => win.show());

  // Open DevTools in dev mode
  if (isDev) {
    win.webContents.openDevTools({ mode: 'detach' });
  }

  // Open external links in the OS browser, not in Electron
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  return win;
}

// ── App lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(() => {
  buildMenu();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// ── IPC: renderer can ask for app version ────────────────────────────────────
ipcMain.handle('app-version', () => app.getVersion());

// ── Menu ──────────────────────────────────────────────────────────────────────
function buildMenu() {
  const template = [
    {
      label: 'CAINE',
      submenu: [
        { label: 'About CAINE Mission Control', role: 'about' },
        { type: 'separator' },
        { label: 'Quit', accelerator: 'CmdOrCtrl+Q', role: 'quit' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { label: 'Reload', accelerator: 'CmdOrCtrl+R', role: 'reload' },
        { label: 'Force Reload', accelerator: 'CmdOrCtrl+Shift+R', role: 'forceReload' },
        { type: 'separator' },
        { label: 'Toggle DevTools', accelerator: 'F12', role: 'toggleDevTools' },
        { type: 'separator' },
        { label: 'Actual Size',  accelerator: 'CmdOrCtrl+0',       role: 'resetZoom' },
        { label: 'Zoom In',      accelerator: 'CmdOrCtrl+Plus',     role: 'zoomIn' },
        { label: 'Zoom Out',     accelerator: 'CmdOrCtrl+Minus',    role: 'zoomOut' },
        { type: 'separator' },
        { label: 'Toggle Fullscreen', accelerator: 'F11', role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Window',
      submenu: [
        { label: 'Minimize', role: 'minimize' },
        { label: 'Zoom',     role: 'zoom' },
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}
