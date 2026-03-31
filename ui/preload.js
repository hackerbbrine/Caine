'use strict';

const { contextBridge, ipcRenderer } = require('electron');

// Expose a minimal, safe API to the renderer via window.caine
contextBridge.exposeInMainWorld('caine', {
  // App version
  version: () => ipcRenderer.invoke('app-version'),
});
