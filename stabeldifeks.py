##Cell1
# @title ⬅️ Jalankan Cell Ini Dulu Untuk Setup
# Langkah 1: Install semua kebutuhan
!pip install diffusers transformers accelerate tqdm

# Langkah 2: Import semua library di satu tempat
import os
import torch
import random
import json
import time
import hashlib
import diffusers
import transformers
import gc
import io
from contextlib import redirect_stdout
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image, PngImagePlugin
from google.colab import drive
from datetime import datetime
from tqdm.notebook import tqdm

# [MODIFIKASI #10] Ganti path hardcoded dengan form Colab
# Pengguna bisa mengubah path ini sebelum menjalankan cell.
output_directory = "/content/drive/MyDrive/Stable_diff" #@param {type:"string"}

# Langkah 3: Mount Google Drive
drive.mount('/content/drive')

# Langkah 4: Konfigurasi dan load model
model_id = "runwayml/stable-diffusion-v1-5"

# =======================================================================
# MEMORY OPTIMIZATION FUNCTIONS
# =======================================================================

def safe_memory_cleanup():
    """Cleanup memory tanpa menghapus model dari GPU"""
    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        print(f"   🧠 GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def setup_memory_efficient_pipeline(pipe):
    """Setup pipeline untuk memory efficiency"""
    try:
        pipe.enable_attention_slicing()
        print("   ✅ Attention slicing enabled")
    except:
        print("   ⚠️ Attention slicing not available")

    try:
        pipe.enable_memory_efficient_attention()
        print("   ✅ Memory efficient attention enabled")
    except:
        print("   ⚠️ Memory efficient attention not available")

    return pipe

def emergency_memory_cleanup():
    """Emergency cleanup jika terjadi OOM error"""
    print("   🚨 Emergency memory cleanup...")
    torch.cuda.empty_cache()
    gc.collect()

    # Force multiple garbage collections
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

    print("   ✅ Emergency cleanup completed")

# Load pipeline dan pindahkan ke GPU
try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    # Ganti scheduler ke DDIM untuk potensi hasil yang lebih baik/cepat
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Apply memory optimizations
    pipe = setup_memory_efficient_pipeline(pipe)

    print(f"✅ Setup selesai. Model '{model_id}' siap digunakan.")
    print(f"🖼️ Hasil akan disimpan di: {output_directory}")
    print("🧠 Memory optimization applied")
except Exception as e:
    print(f"❌ Gagal memuat model. Pastikan Anda menggunakan GPU runtime. Error: {e}")


##Cell2
# @title ⬅️ Jalankan Cell Ini Untuk Menampilkan Aplikasi Generator
# =======================================================================
# BAGIAN 0: IMPORT & SETUP UI Lanjutan
# =======================================================================
import ipywidgets as widgets
from IPython.display import display, clear_output
import sys
from contextlib import contextmanager

# =======================================================================
# BAGIAN 1A: KELAS DAN FUNGSI HELPER UNTUK LOGGING REAL-TIME
# =======================================================================

class Tee:
    """Objek helper yang membagi output (seperti pipa 'T') ke beberapa tujuan."""
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, text):
        self.stream1.write(text)
        self.stream2.write(text)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

@contextmanager
def real_time_logger(log_path):
    """Context manager untuk mengarahkan stdout ke layar dan file secara bersamaan."""
    original_stdout = sys.stdout
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            tee = Tee(original_stdout, log_file)
            sys.stdout = tee
            yield
    finally:
        sys.stdout = original_stdout

# =======================================================================
# BAGIAN 1B: MEMORY MANAGEMENT & VALIDATION FUNCTIONS
# =======================================================================

def validate_batch_size(jobs_list, max_batch=10):
    """Limit batch size untuk prevent OOM"""
    if len(jobs_list) > max_batch:
        print(f"   ⚠️ Batch size ({len(jobs_list)}) terlalu besar!")
        print(f"   📏 Dibatasi ke {max_batch} jobs untuk stabilitas.")
        return jobs_list[:max_batch]
    return jobs_list

def validate_resolution(width, height):
    """Validate resolusi untuk prevent OOM"""
    if width * height > 1536 * 1536:
        return False, f"Resolusi {width}x{height} terlalu tinggi, max 1536x1536"
    if width % 8 != 0 or height % 8 != 0:
        return False, f"Resolusi harus kelipatan 8"
    return True, "OK"

def manage_preview_memory(preview_images, preview_widgets, max_images=4):
    """Smart preview management dengan memory cleanup"""
    if len(preview_images) >= max_images:
        # Clear preview area
        preview_output_area.clear_output(wait=True)

        # Delete old images and widgets
        del preview_images[:]
        del preview_widgets[:]

        # Safe memory cleanup (tidak hapus model)
        safe_memory_cleanup()

        print(f"   🧹 Preview memory cleaned (kept last {max_images} images)")

        return [], []

    return preview_images, preview_widgets

# =======================================================================
# BAGIAN 1C: FUNGSI INTI (GENERATOR & PARSER)
# =======================================================================

def validate_job_params(job_dict, job_index):
    if 'prompt' not in job_dict or not str(job_dict['prompt']).strip():
        return False, f"❌ ERROR di Job Batch #{job_index+1}: 'prompt' tidak boleh kosong."

    # Validate resolution if provided
    if 'width' in job_dict and 'height' in job_dict:
        is_valid, msg = validate_resolution(job_dict['width'], job_dict['height'])
        if not is_valid:
            return False, f"❌ ERROR di Job Batch #{job_index+1}: {msg}"

    return True, "OK"

def parse_batch_json_input(raw_text):
    job_list = []
    if not raw_text.strip(): return []
    try:
        all_jobs = json.loads(raw_text)
        if not isinstance(all_jobs, list):
            print("   > ❌ ERROR: Input harus berupa array JSON (diawali dengan '[' dan diakhiri dengan ']').")
            return []
        for i, job_dict in enumerate(all_jobs):
            is_valid, msg = validate_job_params(job_dict, i)
            if is_valid: job_list.append(job_dict)
            else: print(f"   > {msg}")
        return job_list
    except Exception as e:
        print(f"   > ❌ ERROR saat mem-parse JSON: {e}")
        return []

def generate_stable_diffusion_image(
    model_pipe, output_dir, prompt, negative_prompt="", seed=None,
    width=1024, height=576, num_inference_steps=35, guidance_scale=8.0, eta=0.0
):
    """Fungsi utama untuk generate, menyimpan, dan mencatat metadata gambar."""
    # Validate resolution
    is_valid, msg = validate_resolution(width, height)
    if not is_valid:
        print(f"   ❌ ERROR: {msg}. Melewati job ini.")
        return None, None, None

    if "v1-5" in model_id:
        original_w, original_h = width, height
        width = width - (width % 8)
        height = height - (height % 8)
        if (width, height) != (original_w, original_h):
            print(f"   ⚠️ PERINGATAN: Resolusi {original_w}x{original_h} disesuaikan menjadi {width}x{height} agar kompatibel (kelipatan 8).")

    if width * height > 1024 * 1024:
        print(f"   ⚠️ PERINGATAN: Resolusi {width}x{height} cukup tinggi. Proses mungkin memakan waktu lebih lama.")

    def get_all_hashes(data):
        return {
            "image_md5": hashlib.md5(data).hexdigest(),
            "image_sha256": hashlib.sha256(data).hexdigest(),
            "image_sha512": hashlib.sha512(data).hexdigest()
        }

    print(f"🎨 Memproses prompt: \"{prompt[:60]}...\"")

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"   > Seed tidak diberikan. Menggunakan seed acak: {seed}")
    else:
        try:
            seed = int(seed)
            print(f"   > Menggunakan seed yang diberikan: {seed}")
        except (ValueError, TypeError):
            original_seed = seed
            seed = random.randint(0, 2**32 - 1)
            print(f"   > ⚠️ PERINGATAN: Seed '{original_seed}' tidak valid. Menggunakan seed acak: {seed}")

    try:
        generator = torch.manual_seed(seed)
        start_time = time.time()

        gen_params = {"prompt": prompt, "generator": generator, "width": width, "height": height, "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale, "eta": eta}
        if negative_prompt and negative_prompt.strip():
            gen_params["negative_prompt"] = negative_prompt
        else:
            print("   > Tidak ada negative prompt yang digunakan.")

        image = model_pipe(**gen_params).images[0]
        generation_duration = time.time() - start_time
        print(f"   > Dihasilkan dalam {generation_duration:.2f} detik.")

        image_path = f"{output_dir}/images/img_{seed}.png"
        json_path = f"{output_dir}/seeds/img_{seed}.json"

        png_metadata = PngImagePlugin.PngInfo()
        png_metadata.add_text("prompt", prompt)
        png_metadata.add_text("negative_prompt", negative_prompt)
        png_metadata.add_text("seed", str(seed))
        png_metadata.add_text("guidance_scale", str(guidance_scale))
        png_metadata.add_text("steps", str(num_inference_steps))

        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG', pnginfo=png_metadata)
        image_data = image_bytes.getvalue()
        hashes = get_all_hashes(image_data)

        with open(image_path, "wb") as f:
            f.write(image_data)

        meta_dict = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": seed, "width": width, "height": height, "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale, "eta": eta, "model": model_pipe.config._name_or_path, "scheduler": model_pipe.scheduler.__class__.__name__, "timestamp": datetime.now().isoformat(), "generation_duration_seconds": round(generation_duration, 2), "hashes": hashes}
        with open(json_path, "w") as f: json.dump(meta_dict, f, indent=2)

        print(f"   ✅ Gambar & metadata disimpan ke ('{image_path}', '{json_path}')")

        # Safe memory cleanup setelah generation
        safe_memory_cleanup()

        return image_path, json_path, image.copy()

    except torch.cuda.OutOfMemoryError:
        print("   ❌ GPU Out of Memory! Mencoba emergency cleanup...")
        emergency_memory_cleanup()
        return None, None, None
    except Exception as e:
        print(f"   ❌ Error saat generate: {e}")
        return None, None, None

# =======================================================================
# BAGIAN 2: MEMBUAT KOMPONEN UI UNTUK SETIAP MODE
# =======================================================================
batch_placeholder_text = """[
  {
    "prompt": "a photorealistic image of a cat wearing a samurai armor",
    "width": 768, "height": 768, "seed": 42
  },
  {
    "prompt": "a beautiful beach at sunset, cinematic, 8k"
  }
]"""
batch_textarea = widgets.Textarea(value='', placeholder=batch_placeholder_text, layout={'width': '95%', 'height': '300px'})
batch_button = widgets.Button(description='Generate Batch', button_style='success', icon='play')
batch_ui = widgets.VBox([widgets.Label('Masukkan beberapa job dalam format JSON Array:'), batch_textarea, batch_button])

normal_prompt_area = widgets.Textarea(placeholder='Masukkan prompt Anda di sini...', layout={'width': '95%', 'height': '150px'})
normal_neg_prompt_area = widgets.Textarea(placeholder='(Opsional) Kosongkan jika tidak ingin menggunakan negative prompt.', layout={'width': '95%', 'height': '80px'})
aspect_ratio_dropdown = widgets.Dropdown(options=['Lanskap (16:9) - 1024x576', 'Potret (9:16) - 576x1024', 'Persegi (1:1) - 768x768', 'Lanskap Klasik (4:3) - 960x720', 'Potret Wide (2:3) - 640x960', 'Ultrawide (21:9) - 1344x576'], value='Lanskap (16:9) - 1024x576', description='Aspek Rasio:')
normal_button = widgets.Button(description='Generate Normal', button_style='success', icon='play')
normal_ui = widgets.VBox([widgets.Label('Prompt:'), normal_prompt_area, widgets.Label('Negative Prompt:'), normal_neg_prompt_area, aspect_ratio_dropdown, normal_button])

adv_prompt_area = widgets.Textarea(placeholder='Masukkan prompt Anda di sini...', layout={'width': '95%', 'height': '150px'})
adv_neg_prompt_area = widgets.Textarea(placeholder='(Opsional) Kosongkan jika tidak ingin menggunakan negative prompt.', layout={'width': '95%', 'height': '80px'})
adv_seed_input = widgets.Text(placeholder='Kosong = acak', description='Seed:')
adv_steps_slider = widgets.IntSlider(value=35, min=10, max=60, step=1, description='Steps:')
adv_guidance_slider = widgets.FloatSlider(value=8.0, min=1.0, max=15.0, step=0.5, description='Guidance:')
adv_width_slider = widgets.IntSlider(value=1024, min=256, max=1536, step=8, description='Width:')
adv_height_slider = widgets.IntSlider(value=576, min=256, max=1536, step=8, description='Height:')
adv_button = widgets.Button(description='Generate Advanced', button_style='success', icon='play')
adv_ui = widgets.VBox([widgets.Label('Prompt:'), adv_prompt_area, widgets.Label('Negative Prompt:'), adv_neg_prompt_area, widgets.HBox([adv_seed_input, adv_steps_slider, adv_guidance_slider]), widgets.HBox([adv_width_slider, adv_height_slider]), adv_button])

upload_button = widgets.FileUpload(accept='.json', multiple=False, description='Pilih File JSON')
upload_generate_button = widgets.Button(description='Generate dari JSON', button_style='success', icon='play')
upload_ui = widgets.VBox([widgets.Label('Upload file .json yang berisi array of jobs:'), upload_button, upload_generate_button])

all_buttons = [batch_button, normal_button, adv_button, upload_generate_button]

# =======================================================================
# BAGIAN 3: FUNGSI LOGIKA & EKSEKUSI
# =======================================================================
text_output_area = widgets.Output()
preview_output_area = widgets.Output()
preview_images = []
preview_widgets = []

def execute_generation(jobs_list, mode_name, output_dir, preview_enabled):
    global preview_images, preview_widgets

    for button in all_buttons: button.disabled = True
    preview_toggle.disabled = True

    text_output_area.clear_output(wait=True)

    try:
        with text_output_area:
            log_dir = os.path.join(output_dir, "logging")
            os.makedirs(log_dir, exist_ok=True)
            log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            log_path = os.path.join(log_dir, log_filename)

            with real_time_logger(log_path):
                run_execution_loop(jobs_list, mode_name, output_dir, preview_enabled)
                print(f"\n📝 Log sesi ini disimpan di: {log_path}")

    finally:
        for button in all_buttons: button.disabled = False
        preview_toggle.disabled = False
        if upload_button.value:
            upload_button.value.clear()
            upload_button._counter = 0

def run_execution_loop(jobs_list, mode_name, output_dir, preview_enabled):
    global preview_images, preview_widgets

    preview_output_area.clear_output()
    del preview_images[:], preview_widgets[:]
    gc.collect()

    try:
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/seeds", exist_ok=True)
        negative_prompt_default = "blurry, low quality, jpeg artifacts, distorted, watermark, text, signature"

        if not jobs_list:
            print("\n⚠️ Tidak ada pekerjaan yang valid ditemukan. Proses dihentikan.")
            return

        # Apply batch size limit
        jobs_list = validate_batch_size(jobs_list, max_batch=8)  # Reduced for stability

        total_jobs = len(jobs_list)
        print(f"\n🚀 Memulai proses generate {mode_name}... Total {total_jobs} gambar.")
        if preview_enabled:
            print("   > Tampilan Preview: AKTIF")

        successful_jobs, failed_jobs = 0, 0

        job_iterator = jobs_list
        spinner = ['/', '-', '\\', '|']
        if total_jobs > 1:
            job_iterator = tqdm(jobs_list, desc=f"Memproses {mode_name} jobs")

        for i, job_config in enumerate(job_iterator):
            if total_jobs > 1:
                job_iterator.set_postfix_str(f"Aktivitas: {spinner[i % 4]}")
                print(f"\n--- Memproses Job {i+1}/{total_jobs} ---")

            if 'negative_prompt' not in job_config:
                job_config['negative_prompt'] = negative_prompt_default

            image_object = None
            try:
                _, _, image_object = generate_stable_diffusion_image(model_pipe=pipe, output_dir=output_dir, **job_config)
                if image_object is not None:
                    successful_jobs += 1
                else:
                    failed_jobs += 1
                    continue
            except Exception as e:
                failed_jobs += 1
                print(f"   ❌ FATAL ERROR pada Job #{i+1}: {e}")
                if preview_enabled:
                    with preview_output_area:
                         print(f"Job #{i+1} Gagal, preview tidak dapat ditampilkan.")
                continue

            if preview_enabled and image_object:
                preview_images, preview_widgets = manage_preview_memory(
                    preview_images, preview_widgets, max_images=4
                )

                preview_images.append(image_object)
                img_widget = widgets.Image(value=image_object._repr_png_(), format='png', width=256)
                preview_widgets.append(img_widget)

                with preview_output_area:
                    clear_output(wait=True)
                    display(widgets.HBox(preview_widgets))

        print(f"\n🎉 Proses generate {mode_name} selesai!")
        print(f"   Hasil: {successful_jobs} sukses, {failed_jobs} gagal.")

        # Final memory cleanup
        safe_memory_cleanup()

    except Exception as e:
        print(f"\n❌ Terjadi error tak terduga pada proses eksekusi: {e}")
        emergency_memory_cleanup()

def on_batch_button_clicked(b):
    with text_output_area:
        clear_output(wait=True)
        print("⚙️ Mem-parsing input dari Mode Batch...")
        jobs = parse_batch_json_input(batch_textarea.value)
        if jobs: print(f"✅ Berhasil mem-parsing {len(jobs)} job untuk di-generate.")
    execute_generation(jobs, mode_name="Batch", output_dir=output_directory, preview_enabled=preview_toggle.value)
def on_normal_button_clicked(b):
    with text_output_area:
        clear_output(wait=True)
        print("⚙️ Mempersiapkan job dari Mode Normal...")
        aspect_map = {'Lanskap (16:9) - 1024x576': (1024, 576), 'Potret (9:16) - 576x1024': (576, 1024), 'Persegi (1:1) - 768x768': (768, 768), 'Lanskap Klasik (4:3) - 960x720': (960, 720), 'Potret Wide (2:3) - 640x960': (640, 960), 'Ultrawide (21:9) - 1344x576': (1344, 576)}
        width, height = aspect_map.get(aspect_ratio_dropdown.value, (1024, 576))
        job = {"prompt": normal_prompt_area.value, "negative_prompt": normal_neg_prompt_area.value, "width": width, "height": height}
        if not job["prompt"]: print("❌ Prompt tidak boleh kosong."); return
        print("✅ Berhasil mendeteksi prompt, generate siap dimulai👌")
    execute_generation([job], mode_name="Normal", output_dir=output_directory, preview_enabled=preview_toggle