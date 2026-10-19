"""Automação opcional de interfaces web gratuitas de LLM via Selenium.

⚠️ Uso responsável
------------------
A automação de interfaces web pode violar os Termos de Serviço dos provedores.
Este módulo existe para fins educacionais e de pesquisa, exige que o usuário
autentique manualmente a sessão e **não** é ativado por padrão
(``llm_providers.<nome>.enabled = false`` em ``config.json``).

Requisitos:

    pip install selenium
    # e um ChromeDriver compatível com o navegador instalado

O fluxo é: abrir o navegador com um perfil persistente, esperar o usuário fazer
login na primeira execução, enviar o prompt e aguardar a resposta estabilizar.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.utils import ensure_dir, setup_logger, sha256_text, truncate

logger = setup_logger(__name__)


@dataclass
class WebLLMTarget:
    """Seletores e URLs de uma interface web de chat."""

    name: str
    url: str
    input_selectors: tuple[str, ...]
    send_selectors: tuple[str, ...]
    response_selectors: tuple[str, ...]
    login_markers: tuple[str, ...] = ()


TARGETS: dict[str, WebLLMTarget] = {
    "chatgpt_free": WebLLMTarget(
        name="chatgpt_free",
        url="https://chatgpt.com/",
        input_selectors=("#prompt-textarea", "textarea[data-id]", "div[contenteditable='true']"),
        send_selectors=("button[data-testid='send-button']", "button[aria-label*='Send']"),
        response_selectors=("div[data-message-author-role='assistant']", ".markdown"),
        login_markers=("button[data-testid='login-button']", "text:Log in"),
    ),
    "gemini_free": WebLLMTarget(
        name="gemini_free",
        url="https://gemini.google.com/app",
        input_selectors=("div.ql-editor[contenteditable='true']", "textarea"),
        send_selectors=("button.send-button", "button[aria-label*='Send']"),
        response_selectors=("message-content", ".model-response-text"),
        login_markers=("a[href*='accounts.google.com']",),
    ),
    "claude_free": WebLLMTarget(
        name="claude_free",
        url="https://claude.ai/new",
        input_selectors=("div[contenteditable='true'].ProseMirror", "div[contenteditable='true']"),
        send_selectors=("button[aria-label='Send message']", "button[type='submit']"),
        response_selectors=("div.font-claude-message", "div[data-testid='conversation-turn']"),
        login_markers=("a[href*='login']",),
    ),
}


@dataclass
class SeleniumWebAutomation:
    """Automação de uma interface web específica."""

    target: WebLLMTarget
    profile_dir: Path
    headless: bool = True
    response_timeout: int = 120
    stability_seconds: float = 2.5
    poll_interval: float = 0.75
    _driver: Any = field(default=None, repr=False)

    def available(self) -> bool:
        try:
            import selenium  # noqa: F401
        except ImportError:
            return False
        return os.getenv("AI_CIENTISTA_ENABLE_SELENIUM") == "1"

    def start(self) -> None:
        if self._driver is not None:
            return
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        ensure_dir(self.profile_dir)
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-data-dir={self.profile_dir}")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        self._driver = webdriver.Chrome(options=options)
        self._driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        self._driver.get(self.target.url)
        time.sleep(3)

    def login_required(self) -> bool:
        if self._driver is None:
            return True
        source = self._driver.page_source.lower()
        return any(marker.lower().replace("text:", "") in source for marker in self.target.login_markers)

    def _find(self, selectors: tuple[str, ...], timeout: int = 20) -> Any:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        wait = WebDriverWait(self._driver, timeout)
        for selector in selectors:
            try:
                by = By.XPATH if selector.startswith(("text:", "//")) else By.CSS_SELECTOR
                value = selector.split("text:", 1)[1] if selector.startswith("text:") else selector
                if selector.startswith("text:"):
                    by, value = By.XPATH, f"//*[contains(text(), '{value}')]"
                return wait.until(EC.presence_of_element_located((by, value)))
            except TimeoutException:
                continue
        raise TimeoutError(f"Nenhum seletor encontrado: {selectors}")

    def ask(self, prompt: str) -> str:
        """Envia um prompt e devolve a resposta quando ela estabiliza."""
        if self._driver is None:
            self.start()
        if self.login_required():
            raise RuntimeError(
                f"{self.target.name}: login necessário. Rode uma vez com headless=False "
                f"e AI_CIENTISTA_ENABLE_SELENIUM=1 para autenticar no perfil {self.profile_dir}."
            )
        box = self._find(self.target.input_selectors)
        box.click()
        box.send_keys(prompt)
        for selector in self.target.send_selectors:
            try:
                self._find((selector,), timeout=5).click()
                break
            except Exception:  # noqa: BLE001
                continue
        return self._wait_response()

    def _wait_response(self) -> str:
        deadline = time.time() + self.response_timeout
        last, stable_since = "", time.time()
        while time.time() < deadline:
            try:
                nodes = []
                from selenium.webdriver.common.by import By

                for selector in self.target.response_selectors:
                    nodes = self._driver.find_elements(By.CSS_SELECTOR, selector)
                    if nodes:
                        break
                current = nodes[-1].text if nodes else ""
            except Exception:  # noqa: BLE001
                current = last
            if current and current != last:
                last, stable_since = current, time.time()
            elif current and time.time() - stable_since >= self.stability_seconds:
                return last
            time.sleep(self.poll_interval)
        if not last:
            raise TimeoutError(f"{self.target.name}: resposta não chegou em {self.response_timeout}s")
        return last

    def stop(self) -> None:
        if self._driver is not None:
            try:
                self._driver.quit()
            finally:
                self._driver = None


@dataclass
class SeleniumProvider:
    """Provider de LLM que usa um navegador automatizado como backend."""

    target_name: str = "chatgpt_free"
    headless: bool = True
    profile_root: Path = Path.home() / ".ai_cientista" / "profiles"
    name: str = "selenium"

    def __post_init__(self) -> None:
        self.target_name = os.getenv("AI_CIENTISTA_SELENIUM_TARGET", self.target_name)
        self.name = f"selenium:{self.target_name}"
        self._automation: SeleniumWebAutomation | None = None
        self._cache: dict[str, str] = {}

    def available(self) -> bool:
        try:
            import selenium  # noqa: F401
        except ImportError:
            return False
        return os.getenv("AI_CIENTISTA_ENABLE_SELENIUM") == "1" and self.target_name in TARGETS

    def complete(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        key = sha256_text(full_prompt)
        if key in self._cache:
            return self._cache[key]
        if self._automation is None:
            self._automation = SeleniumWebAutomation(
                target=TARGETS[self.target_name],
                profile_dir=self.profile_root / self.target_name,
                headless=self.headless,
            )
        text = self._automation.ask(full_prompt)
        self._cache[key] = text
        logger.info("Selenium[%s] respondeu %s", self.target_name, truncate(text, 80))
        return text

    def stop(self) -> None:
        if self._automation:
            self._automation.stop()


@dataclass
class SeleniumLLMOrchestrator:
    """Tenta cada interface web em ordem até obter uma resposta."""

    order: tuple[str, ...] = ("chatgpt_free", "gemini_free", "claude_free")

    def __post_init__(self) -> None:
        self.providers = [SeleniumProvider(target_name=name) for name in self.order]

    def complete(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        errors = []
        for provider in self.providers:
            if not provider.available():
                errors.append(f"{provider.name}: indisponível")
                continue
            try:
                return provider.complete(prompt, system=system, **kwargs)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider.name}: {exc}")
        raise RuntimeError("Nenhuma interface web disponível: " + "; ".join(errors))

    def stop(self) -> None:
        for provider in self.providers:
            provider.stop()
