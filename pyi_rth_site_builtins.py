# PyInstaller runtime hook
# ------------------------------------------------------------------
# Normal (paketlenmemiş) Python başlarken `site.py`, `site.sethelper()`
# ve ilgili çağrılarla `help`, `copyright`, `credits`, `license`, `exit`,
# `quit` isimlerini __builtins__ içine ekler. PyInstaller'ın ürettiği
# frozen (--onedir/--onefile) uygulamalarda bu adım çalışmaz; bu yüzden
# import zincirinde bir yerde bu isimlerden biri (ör. `help(...)`)
# kullanılırsa `NameError: name 'help' is not defined` alınır.
#
# Bu hata, main.py'deki genel `except Exception` bloğu tarafından
# yakalanıp "rvc-python bulunamadı" gibi yanıltıcı bir mesaja
# dönüştürülüyordu. Asıl sebep rvc-python'ın eksik olması değil,
# bu builtin'lerin frozen ortamda hiç var olmamasıydı.
#
# Bu dosya, ana script (main.py) çalışmaya başlamadan HEMEN ÖNCE
# PyInstaller bootloader'ı tarafından otomatik olarak import edilir
# (build.yml içinde --runtime-hook ile bildirilmiştir).

import builtins
import sys

if not hasattr(builtins, "help"):
    try:
        import pydoc
        builtins.help = pydoc.help
    except Exception:
        pass

if not hasattr(builtins, "quit") or not hasattr(builtins, "exit"):
    try:
        import _sitebuiltins
        eof = "Ctrl-Z plus Return" if sys.platform == "win32" else "Ctrl-D (i.e. EOF)"
        if not hasattr(builtins, "quit"):
            builtins.quit = _sitebuiltins.Quitter("quit", eof)
        if not hasattr(builtins, "exit"):
            builtins.exit = _sitebuiltins.Quitter("exit", eof)
    except Exception:
        pass

if not hasattr(builtins, "copyright") or not hasattr(builtins, "credits") or not hasattr(builtins, "license"):
    try:
        import _sitebuiltins
        if not hasattr(builtins, "copyright"):
            builtins.copyright = _sitebuiltins._Printer("copyright", sys.copyright)
        if not hasattr(builtins, "credits"):
            builtins.credits = _sitebuiltins._Printer(
                "credits",
                "Python ve genişletmeleri için katkıda bulunanlara teşekkürler.",
            )
        if not hasattr(builtins, "license"):
            builtins.license = _sitebuiltins._Printer(
                "license",
                "Lisans bilgisi için https://docs.python.org/license.html adresine bakın.",
            )
    except Exception:
        pass
