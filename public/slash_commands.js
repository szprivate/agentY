(function () {
  'use strict';

  const COMMANDS = [
    { name: '/restart',       description: 'Restart the agent pipeline' },
    { name: '/stop',          description: 'Stop and shut down the agent' },
    { name: '/unload',        description: 'Unload Ollama models from VRAM' },
    { name: '/clear_vram',    description: 'Clear ComfyUI GPU VRAM' },
    { name: '/clearhistory',  description: 'Delete all conversation history' },
    { name: '/switch_model',  description: 'Switch agent LLM — usage: /switch_model <agent> <provider,model>' },
    { name: '/add_workflow',  description: 'Add a ComfyUI workflow — usage: /add_workflow <path/to/workflow.json>' },
    { name: '/remove_workflow', description: 'Remove a workflow by name — usage: /remove_workflow <template_name>' },
  ];

  let popup = null;
  let selectedIndex = 0;
  let currentInput = null;
  let filteredCommands = [];

  // ── Message history ──────────────────────────────────────────────────────────
  // Keeps up to MAX_HISTORY user messages in localStorage so they survive page
  // reloads, tab closes, and agent restarts.  Arrow-up / Arrow-down in the chat
  // input walks through history (bash-style).

  const MAX_HISTORY = 200;
  const HISTORY_KEY = 'agentY_msgHistory';
  let messageHistory = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  let historyIndex = -1;   // -1 = not browsing history
  let draftValue = '';     // saves the live draft when the user starts browsing

  function pushHistory(msg) {
    msg = (msg || '').trim();
    if (!msg) return;
    if (messageHistory[messageHistory.length - 1] === msg) return; // no dupes
    messageHistory.push(msg);
    if (messageHistory.length > MAX_HISTORY) messageHistory.shift();
    localStorage.setItem(HISTORY_KEY, JSON.stringify(messageHistory));
    historyIndex = -1;
    draftValue = '';
  }

  function getChatInput() {
    return document.getElementById('chat-input') || document.querySelector('textarea');
  }

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

    div.addEventListener('mousedown', function (e) {
      e.preventDefault();
      var item = e.target.closest('.slash-cmd-item');
      if (item) {
        var idx = parseInt(item.dataset.index, 10);
        selectCommand(filteredCommands[idx]);
      }
    });

    div.addEventListener('mouseover', function (e) {
      var item = e.target.closest('.slash-cmd-item');
      if (item) {
        var idx = parseInt(item.dataset.index, 10);
        if (idx !== selectedIndex) {
          selectedIndex = idx;
          renderPopup();
          positionPopupAt(null);
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

  // anchor: DOM element to position relative to; falls back to currentInput
  function positionPopupAt(anchor) {
    var el = anchor || currentInput;
    if (!el) return;
    const p = getPopup();
    const rect = el.getBoundingClientRect();
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

  function showPopup(query, anchor) {
    filteredCommands = query
      ? COMMANDS.filter(function (c) { return c.name.slice(1).startsWith(query); })
      : COMMANDS.slice();
    if (filteredCommands.length === 0) { hidePopup(); return; }
    selectedIndex = 0;
    const p = getPopup();
    p.style.display = 'block';
    renderPopup();
    positionPopupAt(anchor || null);
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
    var needsArgs = ['switch_model', 'add_workflow', 'remove_workflow'].some(function (s) { return cmd.name.indexOf(s) !== -1; });
    var value = needsArgs ? cmd.name + ' ' : cmd.name;
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
    var popupVisible = p.style.display !== 'none';

    // ── Slash-command popup navigation ───────────────────────────────────────
    if (popupVisible) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedIndex = (selectedIndex + 1) % filteredCommands.length;
        renderPopup(); positionPopupAt(null);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedIndex = (selectedIndex - 1 + filteredCommands.length) % filteredCommands.length;
        renderPopup(); positionPopupAt(null);
      } else if (e.key === 'Tab') {
        if (filteredCommands.length > 0) { e.preventDefault(); selectCommand(filteredCommands[selectedIndex]); }
      } else if (e.key === 'Escape') {
        hidePopup();
      }
      return;
    }

    // ── Message history navigation (popup closed) ─────────────────────────────
    var textarea = e.target;
    if (e.key === 'ArrowUp') {
      if (messageHistory.length === 0) return;
      // Allow on first line of textarea (single-line feel)
      var firstNL = textarea.value.indexOf('\n');
      var onFirstLine = firstNL === -1 || textarea.selectionStart <= firstNL;
      if (!onFirstLine) return;
      e.preventDefault();
      if (historyIndex === -1) {
        draftValue = textarea.value;
        historyIndex = messageHistory.length - 1;
      } else if (historyIndex > 0) {
        historyIndex--;
      }
      setReactInputValue(textarea, messageHistory[historyIndex]);
      setTimeout(function () { textarea.selectionStart = textarea.selectionEnd = textarea.value.length; }, 0);
    } else if (e.key === 'ArrowDown') {
      if (historyIndex === -1) return;
      e.preventDefault();
      if (historyIndex < messageHistory.length - 1) {
        historyIndex++;
        setReactInputValue(textarea, messageHistory[historyIndex]);
      } else {
        historyIndex = -1;
        setReactInputValue(textarea, draftValue);
      }
      setTimeout(function () { textarea.selectionStart = textarea.selectionEnd = textarea.value.length; }, 0);
    } else if (e.key !== 'Shift' && e.key !== 'Control' && e.key !== 'Alt' && e.key !== 'Meta') {
      // Any printable key resets browsing so next ArrowUp starts from newest
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') historyIndex = -1;
    }
  }

  function attachToInput(textarea) {
    if (textarea._slashCmdAttached) return;
    textarea._slashCmdAttached = true;
    currentInput = textarea;
    textarea.addEventListener('input', handleInput);
    textarea.addEventListener('keydown', handleKeydown, true);
    textarea.addEventListener('blur', function () { setTimeout(hidePopup, 200); });

    // ── Capture sent message via Enter key (capture phase, before React) ─────
    // Send-button clicks and IME-composed Enter are handled separately below.
    textarea.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        pushHistory(textarea.value);
      }
    }, true); // capture phase fires before React's own handler
  }

  // ── Hook the Send button so clicks (not just Enter) populate history ─────────
  function attachToSendButton(btn) {
    if (!btn || btn._slashCmdHistoryHooked) return;
    btn._slashCmdHistoryHooked = true;
    btn.addEventListener('click', function () {
      var ta = getChatInput();
      if (ta) pushHistory(ta.value);
    }, true); // capture phase, before React's own click handler clears the input
  }

  // ── "/" toolbar button ───────────────────────────────────────────────────────

  function findFileUploadEl() {
    // Scope the search to the chat input area so we don't accidentally pick up
    // a file input that belongs to a different part of the page.
    var ta = document.querySelector('textarea');
    var scope = ta
      ? (ta.closest('form') || ta.closest('[class*="input"]') || ta.closest('[class*="chat"]') || ta.parentElement || document.body)
      : document.body;

    var inp = scope.querySelector('input[type="file"]') || document.querySelector('input[type="file"]');
    if (inp) {
      var label = inp.closest('label');
      if (label) return label;
      var btn = inp.closest('button,[role="button"]');
      if (btn) return btn;
      if (inp.parentElement) return inp.parentElement;
    }
    // Fallbacks by common Chainlit attribute patterns
    return (
      document.querySelector('label[for^="cl-upload"]') ||
      document.querySelector('label[for*="upload"]') ||
      null
    );
  }

  function injectSlashButton() {
    if (document.getElementById('slash-cmd-btn')) return;
    var anchor = findFileUploadEl();
    if (!anchor) return;

    var btn = document.createElement('button');
    btn.id = 'slash-cmd-btn';
    btn.type = 'button';
    btn.title = 'Slash commands';
    btn.textContent = '/';
    Object.assign(btn.style, {
      background: 'none',
      border: 'none',
      cursor: 'pointer',
      color: '#ffffff',
      fontFamily: 'monospace',
      fontSize: '17px',
      fontWeight: '700',
      // Match the size/padding of Chainlit's own icon buttons
      width: '32px',
      height: '32px',
      padding: '0',
      margin: '0 2px',
      borderRadius: '6px',
      lineHeight: '1',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      verticalAlign: 'middle',
      flexShrink: '0',
      transition: 'background 0.15s',
    });

    btn.addEventListener('mouseenter', function () { btn.style.background = '#2d2d50'; });
    btn.addEventListener('mouseleave', function () { btn.style.background = 'none'; });

    btn.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      var p = getPopup();
      if (p.style.display !== 'none') {
        hidePopup();
        return;
      }
      var textarea = document.querySelector('textarea');
      if (textarea) currentInput = textarea;
      showPopup('', btn);
    });

    // Insert immediately AFTER the file-upload element
    anchor.parentNode.insertBefore(btn, anchor.nextSibling);
  }

  // ── Bootstrap ────────────────────────────────────────────────────────────────

  function findAndAttach() {
    var ta = getChatInput();
    if (ta) attachToInput(ta);
    attachToSendButton(document.getElementById('chat-submit'));
    injectSlashButton();
  }

  var observer = new MutationObserver(findAndAttach);
  observer.observe(document.body, { childList: true, subtree: true });
  findAndAttach();
})();
