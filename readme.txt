DeepSeek Admin Paneli Kurulum ve Çalıştırma Kılavuzu (L520 Sunucu - codeserver Ortamı)Bu kılavuz, projenin L520 cihazınızdaki codeserver konteynırı içinde sıfırdan kurulup 7/24 çalıştırılması için hazırlanmıştır. Kurulum, Python'un kurulu olmadığı varsayımıyla başlar ve tmux kullanımı ile codeserver port yönlendirme özelliklerini içerir.Proje Bileşenlerialpha_arena_deepseek.py (Backend Bot): Piyasayı analiz eden, AI'a soran, işlemleri yürüten ve log dosyalarını (portfolio_state.json, trade_history.json, cycle_log.json) oluşturan ana bot.admin_server.py (Web Sunucusu & API): index.html'i sunan ve manuel müdahale komutlarını alan Flask sunucusu.index.html (Frontend): Botun durumunu gösteren web arayüzü.1. Adım: İlk Kurulum (Python, Sanal Ortam ve Kütüphaneler)Bu adımlar sadece ilk kurulumda veya ortam sıfırlandığında gereklidir.Python Kurulumu: codeserver terminalini açın ve Python 3 ile venv modülünü kurun:# Paket listesini güncelle (Debian/Ubuntu tabanlı sistemler için)
sudo apt-get update 

# Python 3, pip ve venv modülünü kur
sudo apt-get install -y python3 python3-pip python3-venv 

# Kurulumu doğrula
python3 --version 
pip --version
Sanal Ortam Oluşturma: Proje dizininize gidin (cd /project/nof1ai vb.) ve sanal ortamı oluşturun:python3 -m venv .venv
Kütüphaneleri Yükleme: Önce sanal ortamı aktif edin, sonra kütüphaneleri yükleyin:# Sanal ortamı aktif et
source .venv/bin/activate 

# Gerekli kütüphaneleri yükle
pip install requests pandas numpy python-dotenv Flask
(Artık terminalinizin başında (.venv) görmelisiniz. Kütüphaneler bu ortama kuruldu.)2. Adım: .env Dosyası Kontrolü.env dosyanızın proje ana dizininde olduğundan ve DeepSeek API anahtarınızın doğru şekilde tanımlandığından emin olun:DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx...
3. Adım: tmux ile Projeyi Başlatma (7/24 Çalışma İçin)Projenin hem botunun hem de sunucusunun sürekli çalışması için tmux kullanacağız. Bu, siz codeserver bağlantısını kesseniz bile işlemlerin arka planda devam etmesini sağlar.Mevcut tmux Oturumlarını Temizleme (Opsiyonel ama Önerilir): Temiz bir başlangıç için eski oturumları kapatın:# Aktif oturumları listele
tmux ls 

# Eğer varsa, örneğin 'arena_bot' adındaki oturumu kapat
# tmux kill-session -t arena_bot 
Yeni tmux Oturumunu Başlatma: Projemiz için yeni bir oturum başlatalım:# 'arena_bot' adında yeni bir oturum başlatır
tmux new -s arena_bot 
(Bu komut sizi otomatik olarak tmux içine alır ve Pencere 0'ı açar.)Botu Çalıştırma (Pencere 0): Bu ilk pencerede botu başlatın:# Sanal ortamı aktif et (tmux penceresi için tekrar gerekir)
source .venv/bin/activate 

# Botu çalıştır
python3 alpha_arena_deepseek.py 
Bot logları bu pencerede akmaya başlayacaktır.Sunucu İçin Yeni Pencere Açma (Pencere 1): Bot çalışırken, aynı tmux oturumu içinde yeni bir terminal penceresi açın:Kısayol: Ctrl+B tuşlarına basın (bırakın), hemen ardından C tuşuna basın.(Artık yeni bir boş terminal penceresindesiniz: Pencere 1)Sunucuyu Çalıştırma (Pencere 1): Bu yeni pencerede web sunucusunu başlatın:# Sanal ortamı bu pencere için de aktif et
source .venv/bin/activate 

# Sunucuyu çalıştır
python3 admin_server.py 
Sunucu 0.0.0.0:8000 adresinde çalışmaya başlayacaktır.Oturumdan Ayrılma (Arka Plana Alma): Her iki işlem de çalışırken, tmux oturumundan güvenle ayrılın:Kısayol: Ctrl+B tuşlarına basın (bırakın), hemen ardından D tuşuna basın.(Terminalde [detached (from session arena_bot)] mesajını görmelisiniz. Artık codeserver sekmesini kapatabilirsiniz, işlemler arka planda devam edecektir.)4. Adım: Arayüze Erişme (codeserver Port Yönlendirme)Bot ve sunucu tmux içinde arka planda çalışırken, arayüze erişmek için:codeserver arayüzünü tarayıcıda açın.Alt paneldeki "PORTS" (PORTLAR) sekmesine gidin. (Eğer görünmüyorsa Ctrl+Shift+P -> Ports: Focus on Ports View komutunu deneyin.)Listede 8000 portunu bulun."Forwarded Address" (Yönlendirilen Adres) veya "Browser" sütunundaki linke tıklayarak arayüzü yeni bir sekmede açın. Bu link http://localhost:8000'den farklıdır.Arayüz açıldığında, birkaç saniye içinde "LIVE" (CANLI) durumuna geçmelidir.5. Adım: tmux Oturumunu Yönetme (Geri Dönme, Durdurma)Oturuma Geri BağlanmaArka planda çalışan botun ve sunucunun loglarını kontrol etmek veya işlemleri durdurmak için tmux oturumuna geri bağlanın:tmux attach -t arena_bot
Pencereler Arası GeçişOturum içindeyken pencereler arasında geçiş yapmak için:Ctrl+B sonra 0 : Botun çalıştığı Pencere 0'a gider.Ctrl+B sonra 1 : Sunucunun çalıştığı Pencere 1'e gider.Ctrl+B sonra sağ/sol ok tuşu: Sıradaki/önceki pencereye gider.Projeyi Güvenli Durdurmatmux attach -t arena_bot ile oturuma bağlanın.Sunucuyu Durdurun: Pencere 1'e geçin (Ctrl+B 1) ve Ctrl+C tuşlarına basın.Botu Durdurun: Pencere 0'a geçin (Ctrl+B 0) ve Ctrl+C tuşlarına basın.tmux Oturumunu Kapatın: Her iki işlem durduktan sonra, her iki pencerede de exit yazarak tmux oturumunu tamamen sonlandırabilirsiniz.6. Adım: Sorun GidermeSorunÇözümArayüz "DISCONNECTED"admin_server.py (Pencere 1) ve alpha_arena_deepseek.py (Pencere 0) işlemlerinin tmux içinde çalıştığından emin olun. Hata mesajı var mı kontrol edin."Port already in use"admin_server.py dosyasında PORT = 8000 değerini değiştirin (örn: 8001). tmux oturumunu kapatıp (tmux kill-session -t arena_bot) tekrar başlatın. Arayüze yeni porttan erişin.ModuleNotFoundErrorİlgili terminalde source .venv/bin/activate komutunu çalıştırdığınızdan ve pip install requests pandas... komutunu bu sanal ortam aktifken çalıştırdığınızdan emin olun. Gerekirse kütüphaneleri tekrar yükleyin.tmux komutları çalışmıyorsudo apt install tmux komutu ile tmux'ın kurulu olduğundan emin olun.sudo apt-get update komutu çalışmıyor veya hata veriyorKonteyneriniz farklı bir Linux dağıtımı (örn: Alpine) kullanıyor olabilir. Alpine için apk update && apk add python3 py3-pip py3-venv tmux komutunu deneyin. Dağıtımınızı cat /etc/os-release komutu ile kontrol edebilirsiniz.