(function () {
  'use strict';

  const COMMANDS = [
    { name: '/restart',      description: 'Restart the agent pipeline' },
    { name: '/stop',         description: 'Stop and shut down the agent' },
    { name: '/unload',       description: 'Unload Ollama models from VRAM' },
    { name: '/clear_vram',   description: 'Clear ComfyUI GPU VRAM' },
    { name: '/clearhistory', description: 'Delete all conversation history' },
    { name: '/switch_model', description: 'Switch agent LLM — usage: /switch_model <agent> <provider,model>' },
  ];

  let popup = null;
  let selectedIndex = 0;
  let currentInput = null;
  let filteredCommands = [];

  // ── Popup DOM ────────────────────────────────────────────────────────────────

  function createPopup() {
    const div = document.createElement('div');
    div.id = 'slash-command-popup';
    Object.assign(div.style, {
      position: 'fixed',
      background: '#1e1e2e',
      border: '1px solid #3a3a5c',
      borderRadius: '8px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
      zIndex: '99999',
      minWidth: '320px',
      maxWidth: '480px',
      overflow: 'hidden',
      display: 'none',
      fontFamily: 'system-ui, -apple-system, sans-serif',
    });

    // Single delegated mousedown on the container — survives innerHTML re-renders
    div.addEventListener('mousedown', function (e) {
      e.preventDefault(); // keep textarea focused
      var item = e.target.closest('.slash-cmd-item');
      if (item) {
        var idx = parseInt(item.dataset.index, 10);
        selectCommand(filteredCommands[idx]);
      }
    });

    // Hover via mouseover delegation — no per-item listeners needed
    div.addEventListener('mouseover', function (e) {
      var item = e.target.closest('.slash-cmd-item');
      if (item) {
        var idx = parseInt(item.dataset.index, 10);
        if (idx !== selectedIndex) {
          selectedIndex = idx;
          renderPopup();
          positionPopup();
        }
      }
    });

    document.body.appendChild(div);
    return div;
  }

  function getPopup() {
    if (!popup) popup = createPopup();
    return popup;
  }

  // ── Rendering ────────────────────────────────────────────────────────────────

  function renderPopup() {
    const p = getPopup();
    p.innerHTML =
      '<div style="padding:6px 12px 4px;font-size:11px;color:#666;letter-spacing:.05em;text-transform:uppercase;">Commands</div>' +
      filteredCommands.map(function (cmd, i) {
        const sel = i === selectedIndex;
        return (
          '<div class="slash-cmd-item" data-index="' + i + '" style="' +
          'padding:8px 12px;cursor:pointer;display:flex;align-items:center;gap:12px;' +
          'background:' + (sel ? '#2d2d50' : 'transparent') + ';' +
          'border-left:3px solid ' + (sel ? '#7c83ff' : 'transparent') + ';' +
          '">' +
          '<span style="font-family:monospace;font-size:13px;font-weight:600;' +
          'color:' + (sel ? '#9da5ff' : '#7c83ff') + ';min-width:130px;">' + cmd.name + '</span>' +
          '<span style="font-size:12px;color:#888;">' + cmd.description + '</span>' +
          '</div>'
        );
      }).join('');
  }

  function positionPopup() {
    if (!currentInput) return;
    const p = getPopup();
    const rect = currentInput.getBoundingClientRect();
    const popupH = p.offsetHeight || filteredCommands.length * 40 + 30;
    if (rect.top > popupH || rect.top > window.innerHeight - rect.bottom) {
      p.style.bottom = (window.innerHeight - rect.top + 6) + 'px';
      p.style.top = 'auto';
    } else {
      p.style.top = (rect.bottom + 6) + 'px';
      p.style.bottom = 'auto';
    }
    p.style.left = rect.left + 'px';
  }

  function showPopup(query) {
    filteredCommands = query
      ? COMMANDS.filter(function (c) { return c.name.slice(1).startsWith(query); })
      : COMMANDS.slice();
    if (filteredCommands.length === 0) { hidePopup(); return; }
    selectedIndex = 0;
    const p = getPopup();
    p.style.display = 'block';
    renderPopup();
    positionPopup();
  }

  function hidePopup() {
    if (popup) popup.style.display = 'none';
  }

  // ── React value injection ────────────────────────────────────────────────────

  function setReactInputValue(el, value) {
    var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    nativeSetter.call(el, value);
    var tracker = el._valueTracker;
    if (tracker) tracker.setValue('');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function selectCommand(cmd) {
    var textarea = document.querySelector('textarea');
    hidePopup();
    if (!textarea) return;
    textarea.focus();
    // Commands that require arguments get a trailing space so the user can
    // continue typing without manually pressing space first.
    var value = cmd.name.indexOf('switch_model') !== -1 ? cmd.name + ' ' : cmd.name;
    setReactInputValue(textarea, value);
  }

  // ── Input listeners ──────────────────────────────────────────────────────────

  function handleInput(e) {
    var val = e.target.value;
    if (val === '/') {
      showPopup('');
    } else if (val.startsWith('/') && !val.includes(' ')) {
      showPopup(val.slice(1));
    } else {
      hidePopup();
    }
  }

  function handleKeydown(e) {
    var p = getPopup();
    if (p.style.display === 'none') return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = (selectedIndex + 1) % filteredCommands.length;
      renderPopup(); positionPopup();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = (selectedIndex - 1 + filteredCommands.length) % filteredCommands.length;
      renderPopup(); positionPopup();
    } else if (e.key === 'Tab') {
      if (filteredCommands.length > 0) { e.preventDefault(); selectCommand(filteredCommands[selectedIndex]); }
    } else if (e.key === 'Escape') {
      hidePopup();
    }
  }

  function attachToInput(textarea) {
    if (textarea._slashCmdAttached) return;
    textarea._slashCmdAttached = true;
    currentInput = textarea;
    textarea.addEventListener('input', handleInput);
    textarea.addEventListener('keydown', handleKeydown, true);
    textarea.addEventListener('blur', function () { setTimeout(hidePopup, 200); });
  }

  function findAndAttach() {
    var ta = document.querySelector('textarea');
    if (ta) attachToInput(ta);
  }

  var observer = new MutationObserver(findAndAttach);
  observer.observe(document.body, { childList: true, subtree: true });
  findAndAttach();
})();
