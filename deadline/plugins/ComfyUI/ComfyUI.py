#!/usr/bin/env python3
"""Deadline 10 plugin "ComfyUI": run an API-format ComfyUI workflow on a farm worker.

Per job (StartJob): find the ComfyUI install, start a headless ComfyUI on a free port (or reuse a healthy one
left warm by an earlier job), upload the job's input files into it.
Per task (RenderTasks): post the prompt (the job's auxiliary file) to /prompt, poll /history until it is done,
download the outputs (/view) into the job's output directory (a network path), write a small manifest next to them
for agentY, report progress. An execution error fails the task with ComfyUI's own error text.
End of job (EndJob): stop the ComfyUI this plugin started, unless "Keep ComfyUI warm" is on.

A worker without ComfyUI fails the task at once ("ComfyUI not installed on <worker>") so Deadline's bad-worker
detection moves the job on; the proper fix is to keep such workers out of the "comfyui" pool.

Partner / API nodes need the Comfy.org API key. It is read on the worker, from the environment variable named in
the plugin settings (COMFY_ORG_API_KEY by default) or from a key file, and sent as extra_data.api_key_comfy_org.
It is never part of the job files.

Plugin settings (Monitor > Tools > Configure Plugins > ComfyUI): ComfyUIPath_Windows / _Linux / _Mac, PythonExe,
ExtraArgs, StartupTimeoutSeconds, KeepWarm, ApiKeyEnvVar, ApiKeyFile.
Plugin info (written by agentY): PromptFile, OutputDirectory, InputDirectory, Input0..N, SeedMode, ComfyUIPath.
"""

from __future__ import absolute_import

import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from Deadline.Plugins import DeadlinePlugin, PluginType  # pylint: disable=import-error
from Deadline.Scripting import RepositoryUtils, SystemUtils  # pylint: disable=import-error

STATE_FILE = os.path.join(tempfile.gettempdir(), "agenty_comfyui_worker_state.json")
MANIFEST_PREFIX = ".agenty_farm_"
_TQDM = re.compile(r"(\d{1,3})%\|")


def GetDeadlinePlugin():
    return ComfyUIPlugin()


def CleanupDeadlinePlugin(deadlinePlugin):
    deadlinePlugin.Cleanup()


class ComfyUIPlugin(DeadlinePlugin):
    def __init__(self):
        super().__init__()
        self.InitializeProcessCallback += self.InitializeProcess
        self.StartJobCallback += self.StartJob
        self.RenderTasksCallback += self.RenderTasks
        self.EndJobCallback += self.EndJob
        self.proc = None
        self.port = 0
        self.base = ""
        self.log_path = ""
        self.log_pos = 0
        self.owns_server = False
        self.keep_warm = False

    def Cleanup(self):
        del self.InitializeProcessCallback
        del self.StartJobCallback
        del self.RenderTasksCallback
        del self.EndJobCallback
        self._stop_server(force=not self.keep_warm)

    def InitializeProcess(self):
        self.SingleFramesOnly = True
        self.PluginType = PluginType.Advanced

    # -- job ------------------------------------------------------------------

    def StartJob(self):
        host = socket.gethostname()
        self.keep_warm = self.GetBooleanConfigEntryWithDefault("KeepWarm", False)
        root = self._comfy_root()
        if not root or not os.path.isdir(root):
            self.FailRender("ComfyUI not installed on %s: install folder %r does not exist. Set the plugin setting "
                            "ComfyUIPath_%s (or DEADLINE_COMFYUI_PATH at submit), or take this worker out of the "
                            "ComfyUI pool." % (host, root or "(not set)", self._os_name()))
        main = self._main_py(root)
        if not main:
            self.FailRender("ComfyUI not installed on %s: no main.py under %s" % (host, root))
        python = self._python(root)
        if not python:
            self.FailRender("ComfyUI not installed on %s: no Python found for %s (set the plugin setting PythonExe)"
                            % (host, root))
        self._start_server(root, main, python)
        self._upload_inputs()

    def EndJob(self):
        if self.keep_warm:
            self.LogInfo("Keeping ComfyUI warm on port %s" % self.port)
            return
        self._stop_server(force=True)

    # -- one task -------------------------------------------------------------

    def RenderTasks(self):
        frame = self.GetStartFrame()
        prompt = self._load_prompt(frame)
        self._check_alive()
        payload = {"prompt": prompt, "client_id": uuid.uuid4().hex}
        key = self._api_key()
        if key:
            payload["extra_data"] = {"api_key_comfy_org": key}
        self.SetStatusMessage("Queueing prompt")
        try:
            resp = self._http("POST", "/prompt", json.dumps(payload).encode("utf-8"),
                              {"Content-Type": "application/json"})
        except urllib.error.HTTPError as exc:
            self.FailRender("ComfyUI rejected the workflow: %s" % self._describe_reject(exc.read()))
        prompt_id = json.loads(resp).get("prompt_id")
        if not prompt_id:
            self.FailRender("ComfyUI returned no prompt_id: %s" % resp[:400])
        self.LogInfo("Prompt queued: %s" % prompt_id)
        entry = self._wait_for(prompt_id)
        status = entry.get("status", {})
        if status.get("status_str") == "error" or self._exec_error(status):
            self.FailRender(self._exec_error(status) or "ComfyUI reported an error (no details in /history)")
        files = self._collect_outputs(entry.get("outputs", {}), frame)
        self._write_manifest(frame, files)
        self.SetProgress(100)
        self.SetStatusMessage("Done: %d file(s)" % len(files))
        self.LogInfo("Outputs: %s" % ", ".join(files))

    # -- settings -------------------------------------------------------------

    @staticmethod
    def _os_name():
        return "Windows" if sys.platform.startswith("win") else ("Mac" if sys.platform == "darwin" else "Linux")

    def _comfy_root(self):
        v = self.GetConfigEntryWithDefault("ComfyUIPath_" + self._os_name(), "").strip()
        if not v:
            v = self.GetPluginInfoEntryWithDefault("ComfyUIPath", "").strip()   # DEADLINE_COMFYUI_PATH at submit time
        return RepositoryUtils.CheckPathMapping(v) if v else ""

    @staticmethod
    def _main_py(root):
        for rel in ("main.py", os.path.join("ComfyUI", "main.py")):
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                return p
        return ""

    def _python(self, root):
        v = self.GetConfigEntryWithDefault("PythonExe", "").strip()
        if v:
            return v if os.path.isfile(v) else ""
        win = sys.platform.startswith("win")
        sub = ("Scripts", "python.exe") if win else ("bin", "python")
        parent = os.path.dirname(root.rstrip("\\/"))
        for base in (root, parent):
            for cand in (os.path.join(base, "python_embeded", "python.exe"), os.path.join(base, ".venv", *sub),
                         os.path.join(base, "venv", *sub)):
                if os.path.isfile(cand):
                    return cand
        return shutil.which("python") or shutil.which("python3") or ""

    def _api_key(self):
        var = self.GetConfigEntryWithDefault("ApiKeyEnvVar", "COMFY_ORG_API_KEY").strip() or "COMFY_ORG_API_KEY"
        key = os.environ.get(var, "").strip()
        if key:
            return key
        f = self.GetConfigEntryWithDefault("ApiKeyFile", "").strip()
        if f and os.path.isfile(f):
            with open(f, "r") as fh:
                return fh.read().strip()
        return ""

    # -- the ComfyUI process ----------------------------------------------------

    def _start_server(self, root, main, python):
        st = self._read_state()
        if st and st.get("root") == root and self._healthy(st.get("port", 0)):
            self.port, self.owns_server = st["port"], False
            self.base = "http://127.0.0.1:%d" % self.port
            self.LogInfo("Reusing the warm ComfyUI on port %d" % self.port)
            return
        if st and st.get("pid"):
            self._kill(st["pid"])                      # stale or a different install
        self.port = self._free_port()
        self.base = "http://127.0.0.1:%d" % self.port
        cmd = [python, "-s", main, "--listen", "127.0.0.1", "--port", str(self.port), "--disable-auto-launch"]
        extra = self.GetConfigEntryWithDefault("ExtraArgs", "").strip()
        if extra:
            cmd += extra.split()
        self.log_path = os.path.join(tempfile.gettempdir(), "agenty_comfyui_%d.log" % self.port)
        log = open(self.log_path, "wb")
        kw = {}
        if sys.platform.startswith("win"):
            kw["creationflags"] = 0x00000200 | 0x08000000      # new process group, no window
        else:
            kw["start_new_session"] = True
        self.LogInfo("Starting ComfyUI: %s" % " ".join(cmd))
        self.proc = subprocess.Popen(cmd, cwd=os.path.dirname(main), stdout=log, stderr=subprocess.STDOUT, **kw)
        self.owns_server = True
        self._write_state({"pid": self.proc.pid, "port": self.port, "root": root})
        timeout = int(self.GetConfigEntryWithDefault("StartupTimeoutSeconds", "300") or 300)
        t0 = time.time()
        self.SetStatusMessage("Starting ComfyUI")
        while not self._healthy(self.port):
            if self.proc.poll() is not None:
                self.FailRender("ComfyUI exited at startup (code %s). Log tail:\n%s"
                                % (self.proc.returncode, self._log_tail()))
            if self.IsCanceled():
                self.FailRender("Received cancel task command")
            if time.time() - t0 > timeout:
                self._stop_server(force=True)
                self.FailRender("ComfyUI did not answer within %ds. Log tail:\n%s" % (timeout, self._log_tail()))
            SystemUtils.Sleep(1000)
        self.LogInfo("ComfyUI is up on port %d after %.0fs" % (self.port, time.time() - t0))

    def _stop_server(self, force):
        if self.proc is not None and self.owns_server and force:
            self._kill(self.proc.pid)
            try:
                os.remove(STATE_FILE)
            except OSError:
                pass
        self.proc = None

    @staticmethod
    def _kill(pid):
        try:
            if sys.platform.startswith("win"):
                subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            else:
                import signal
                os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _read_state():
        try:
            with open(STATE_FILE, "r") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_state(d):
        try:
            with open(STATE_FILE, "w") as fh:
                json.dump(d, fh)
        except OSError:
            pass

    @staticmethod
    def _free_port():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    @staticmethod
    def _healthy(port):
        if not port:
            return False
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/system_stats" % port, timeout=3).read()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _check_alive(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.FailRender("ComfyUI died (code %s). Log tail:\n%s" % (self.proc.returncode, self._log_tail()))
        if not self._healthy(self.port):
            self.FailRender("ComfyUI on port %d does not answer" % self.port)

    def _log_tail(self, n=2500):
        try:
            with open(self.log_path, "rb") as fh:
                return fh.read()[-n:].decode("utf-8", "replace")
        except OSError:
            return "(no log)"

    def _pump_log(self):
        """Forward new ComfyUI log lines to the task log; return the latest sampler percentage."""
        pct = None
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(self.log_pos)
                data = fh.read()
                self.log_pos = fh.tell()
        except OSError:
            return None
        shown = 0
        for line in re.split(r"[\r\n]+", data.decode("utf-8", "replace")):
            m = _TQDM.search(line)
            if m:
                pct = int(m.group(1))
            elif line.strip() and shown < 25:
                self.LogInfo(line.rstrip()[:400])
                shown += 1
        return pct

    # -- HTTP -------------------------------------------------------------------

    def _http(self, method, path, data=None, headers=None, timeout=60):
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")

    # -- inputs / prompt --------------------------------------------------------

    def _upload_inputs(self):
        src_dir = self.GetPluginInfoEntryWithDefault("InputDirectory", "")
        if not src_dir:
            return
        src_dir = RepositoryUtils.CheckPathMapping(src_dir)
        i = 0
        while True:
            rel = self.GetPluginInfoEntryWithDefault("Input%d" % i, "")
            if not rel:
                break
            i += 1
            path = os.path.join(src_dir, rel)
            if not os.path.isfile(path):
                self.FailRender("Input file missing on the share: %s" % path)
            sub, name = os.path.split(rel.replace("\\", "/"))
            b = "----agentY" + uuid.uuid4().hex
            with open(path, "rb") as fh:
                blob = fh.read()
            parts = []
            for k, v in (("type", "input"), ("subfolder", sub), ("overwrite", "true")):
                parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode())
            parts.append(("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\n"
                          "Content-Type: application/octet-stream\r\n\r\n" % (b, name)).encode() + blob + b"\r\n")
            body = b"".join(parts) + ("--%s--\r\n" % b).encode()
            self.LogInfo("Uploading input %s (%d bytes)" % (rel, len(blob)))
            self._http("POST", "/upload/image", body, {"Content-Type": "multipart/form-data; boundary=" + b},
                       timeout=600)

    def _load_prompt(self, frame):
        try:
            path = self.GetDataFilename()
        except Exception:  # noqa: BLE001
            path = ""
        if not path or not os.path.isfile(path):
            path = os.path.join(self.GetJobsDataDirectory(), self.GetPluginInfoEntryWithDefault("PromptFile", "prompt.json"))
        with open(path, "r", encoding="utf-8") as fh:
            prompt = json.load(fh)
        if self.GetPluginInfoEntryWithDefault("SeedMode", "keep") == "randomize":
            rng = random.Random("%s:%s" % (self.GetJob().JobId, frame))
            n = 0
            for node in prompt.values():
                ins = node.get("inputs", {})
                for k in ("seed", "noise_seed"):
                    if isinstance(ins.get(k), int) and not isinstance(ins.get(k), bool):
                        ins[k] = rng.randint(0, 2 ** 52)
                        n += 1
            self.LogInfo("Randomized %d seed(s) for frame %s" % (n, frame))
        return prompt

    # -- waiting, errors, outputs -------------------------------------------------

    @staticmethod
    def _describe_reject(raw):
        try:
            d = json.loads(raw)
        except ValueError:
            return raw[:600].decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)[:600]
        err = d.get("error", {})
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
        det = err.get("details", "") if isinstance(err, dict) else ""
        nodes = []
        for nid, ne in (d.get("node_errors") or {}).items():
            for e in ne.get("errors", []):
                nodes.append("node %s (%s): %s %s" % (nid, ne.get("class_type", "?"), e.get("message", ""), e.get("details", "")))
        return "; ".join(x for x in [msg, det] + nodes[:6] if x)[:1500]

    @staticmethod
    def _exec_error(status):
        for kind, data in status.get("messages", []):
            if kind == "execution_error":
                return "ComfyUI execution_error in %s (node %s): %s: %s" % (
                    data.get("node_type", "?"), data.get("node_id", "?"), data.get("exception_type", ""),
                    data.get("exception_message", "").strip())
        return ""

    def _wait_for(self, prompt_id):
        t0 = time.time()
        last = -1
        while True:
            if self.IsCanceled():
                try:
                    self._http("POST", "/interrupt", b"{}", {"Content-Type": "application/json"}, timeout=10)
                except Exception:  # noqa: BLE001
                    pass
                self.FailRender("Received cancel task command")
            if self.proc is not None and self.proc.poll() is not None:
                self.FailRender("ComfyUI died during the run (code %s). Log tail:\n%s"
                                % (self.proc.returncode, self._log_tail()))
            pct = self._pump_log()
            if pct is not None and pct != last:
                last = pct
                self.SetProgress(min(pct, 99))
                self.SetStatusMessage("Sampling %d%%" % pct)
            try:
                h = json.loads(self._http("GET", "/history/" + prompt_id, timeout=30))
            except Exception:  # noqa: BLE001
                h = {}
            entry = h.get(prompt_id)
            if entry and (entry.get("status", {}).get("completed") or entry.get("status", {}).get("status_str") in ("success", "error")):
                self._pump_log()
                return entry
            SystemUtils.Sleep(1000)

    def _collect_outputs(self, outputs, frame):
        out_dir = self.GetPluginInfoEntryWithDefault("OutputDirectory", "")
        if not out_dir:
            self.FailRender("No OutputDirectory in the plugin info")
        mapped = RepositoryUtils.CheckPathMapping(out_dir)
        os.makedirs(mapped, exist_ok=True)
        files = []
        for node_out in outputs.values():
            for items in node_out.values():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not (isinstance(it, dict) and it.get("filename")) or it.get("type", "output") != "output":
                        continue
                    stem, ext = os.path.splitext(it["filename"])
                    name = "%s_t%04d%s" % (stem, int(frame), ext)
                    q = urllib.parse.urlencode({"filename": it["filename"], "subfolder": it.get("subfolder", ""), "type": "output"})
                    dst = os.path.join(mapped, name)
                    tmp = dst + ".part"
                    with urllib.request.urlopen(self.base + "/view?" + q, timeout=600) as r, open(tmp, "wb") as fh:
                        shutil.copyfileobj(r, fh)
                    os.replace(tmp, dst)
                    files.append(os.path.join(out_dir, name))
        if not files:
            self.LogWarning("The workflow finished but produced no output files")
        return files

    def _write_manifest(self, frame, files):
        out_dir = RepositoryUtils.CheckPathMapping(self.GetPluginInfoEntryWithDefault("OutputDirectory", ""))
        path = os.path.join(out_dir, "%s%s_%s.json" % (MANIFEST_PREFIX, self.GetJob().JobId, frame))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"job": self.GetJob().JobId, "frame": frame, "files": files}, fh)
