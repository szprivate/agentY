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
    { name: '/resend',        description: 'Resend the first user message of the current thread' },
    { name: '/remove_workflow', description: 'Remove a workflow by name — usage: /remove_workflow <template_name>' }
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
    var anchor = findFileUploadEl();
    if (!anchor) return;

    var existing = document.getElementById('slash-cmd-btn');
    // Re-inject if the button was detached from the DOM or is no longer the
    // immediate next sibling of the attach-files element (React re-render).
    if (existing) {
      if (existing.isConnected && anchor.nextSibling === existing) return;
      existing.remove();
    }

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

  // ── "Resend first message" menu item ─────────────────────────────────────────
  //
  // Injects a "Resend first message" entry into the Radix dropdown that opens
  // from the three-dot button on each thread in the sidebar history.  When
  // clicked, it navigates to the target thread and submits `/resend` there so
  // the backend (chainlit_app.py) replays the thread's original first user
  // message + image attachments as a fresh request inside that same thread.
  //
  // We can't modify the Chainlit react bundle, so we hook the DOM:
  //   1. Track the most recently-pressed thread three-dot button so we know
  //      which thread the menu belongs to (Radix doesn't expose that on the
  //      menu itself).
  //   2. Watch for a Radix dropdown menu opening; if it contains the
  //      Rename/Delete items it's the thread menu — clone one of its items,
  //      rewrite the label, and append "Resend first message".
  // ────────────────────────────────────────────────────────────────────────────

  let lastThreadIdContext = null;

  function findThreadIdForButton(btn) {
    // Walk up looking for a sibling/descendant <a href="/thread/<id>">.
    let cur = btn;
    while (cur && cur !== document.body) {
      var link = cur.querySelector ? cur.querySelector('a[href^="/thread/"]') : null;
      if (link) {
        var m = link.getAttribute('href').match(/\/thread\/([^?#/]+)/);
        if (m) return m[1];
      }
      // Also try the element itself if it's an anchor
      if (cur.tagName === 'A' && cur.getAttribute) {
        var href = cur.getAttribute('href') || '';
        var m2 = href.match(/^\/thread\/([^?#/]+)/);
        if (m2) return m2[1];
      }
      cur = cur.parentElement;
    }
    return null;
  }

  document.addEventListener('mousedown', function (e) {
    var btn = e.target.closest && e.target.closest('button');
    if (!btn) return;
    var tid = findThreadIdForButton(btn);
    if (tid) lastThreadIdContext = tid;
  }, true);

  function delay(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function findThreadAnchor(threadId) {
    return document.querySelector('a[href="/thread/' + threadId + '"]') ||
           document.querySelector('a[href^="/thread/' + threadId + '?"]') ||
           document.querySelector('a[href^="/thread/' + threadId + '#"]');
  }

  async function resendInThread(threadId) {
    if (!threadId) return;
    // Navigate to the target thread if we aren't already there.
    var alreadyThere = location.pathname === '/thread/' + threadId;
    if (!alreadyThere) {
      var anchor = findThreadAnchor(threadId);
      if (anchor) {
        anchor.click();
      } else {
        // Fallback: hard-navigate. Loses session but at least gets us there.
        location.href = '/thread/' + threadId;
        return;
      }
      // Wait for the route to swap and the textarea to remount.
      for (var i = 0; i < 40; i++) {
        await delay(80);
        if (location.pathname === '/thread/' + threadId) break;
      }
    }
    // Re-find the textarea (it may have re-mounted on route change).
    var textarea = null;
    for (var j = 0; j < 40; j++) {
      textarea = getChatInput();
      if (textarea) break;
      await delay(80);
    }
    if (!textarea) return;
    // Give Chainlit a beat to wire up its socket for the resumed thread.
    await delay(150);
    textarea.focus();
    setReactInputValue(textarea, '/resend');
    await delay(60);
    // Submit. Prefer the explicit chat-submit button, fall back to Enter.
    var sendBtn = document.getElementById('chat-submit') ||
                  document.querySelector('button[type="submit"]') ||
                  document.querySelector('button[aria-label*="Send" i]');
    if (sendBtn && !sendBtn.disabled) {
      sendBtn.click();
    } else {
      textarea.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'Enter', code: 'Enter', bubbles: true, cancelable: true,
      }));
    }
  }

  function injectResendIntoMenu(menu) {
    if (!menu || menu._resendInjected) return;
    var items = menu.querySelectorAll('[role="menuitem"]');
    if (items.length === 0) return;
    var labels = Array.from(items).map(function (it) { return (it.textContent || '').trim().toLowerCase(); });
    // Only inject into the thread menu (Rename/Delete present). Avoids polluting
    // unrelated dropdowns like the user-profile menu.
    var isThreadMenu = labels.some(function (t) { return /^rename$/.test(t); }) &&
                       labels.some(function (t) { return /^delete$/.test(t); });
    if (!isThreadMenu) return;
    var tid = lastThreadIdContext;
    if (!tid) return;

    // Clone the last item to inherit Radix styling, then rewrite label & icon.
    var template = items[items.length - 1];
    var newItem = template.cloneNode(true);
    // Strip any svg icon node so the entry shows text only — easier than
    // guessing which lucide-react icon to swap in.
    newItem.querySelectorAll('svg').forEach(function (s) { s.remove(); });
    // Replace text content of all leaf text-bearing children with our label.
    // Easiest: clear and set a single span.
    newItem.innerHTML = '';
    var span = document.createElement('span');
    span.textContent = 'Resend first message';
    newItem.appendChild(span);

    newItem.addEventListener('click', function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      var capturedTid = tid;
      // Close the dropdown so the navigation/click isn't swallowed.
      document.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'Escape', code: 'Escape', bubbles: true,
      }));
      setTimeout(function () { resendInThread(capturedTid); }, 50);
    }, true);

    menu.appendChild(newItem);
    menu._resendInjected = true;
  }

  var menuObserver = new MutationObserver(function () {
    document.querySelectorAll('[role="menu"]').forEach(injectResendIntoMenu);
  });
  menuObserver.observe(document.body, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['data-state'],
  });

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
