import os
import tarfile
import requests

def compile_latex():
    # Detect directories dynamically relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tarball_name = os.path.join(script_dir, "latex_project.tar.bz2")
    main_tex = os.path.join(script_dir, "HO_main2.tex")
    plots_dir = os.path.abspath(os.path.join(script_dir, "..", "plots"))
    
    print("Creating tarball...")
    with tarfile.open(tarball_name, "w:bz2") as tar:
        # Add the main tex file
        if os.path.exists(main_tex):
            tar.add(main_tex, arcname="HO_main2.tex")
            print(f"Added {main_tex} as HO_main2.tex")
        else:
            print(f"Error: {main_tex} not found!")
            return
        
        # File mapping for referenced figures to restrict size and map correct names
        file_mapping = {
            "0_dataset_visualisation.png": "0_dataset_visualisation.png",
            "fig2_method_a.png": "fig2_method_a.png",
            "fig3_method_b.png": "fig3_method_b.png",
            "fig4_method_c.png": "fig4_method_c.png",
            "plots/fig_cd_diagram.png": "cd_diagram.png",
            "plots/fig6_hpobench_regret.png": "fig6_hpobench_regret.png",
            "plots/fig6b_hpolib_regret.png": "fig6b_hpolib_regret.png",
            "plots/fig7_nasbench201_regret.png": "fig7_nasbench201_regret.png",
            "plots/fig8_hpobench_summary.png": "fig8_hpobench_summary.png",
            "plots/fig5_comparative.png": "fig5_comparative.png",
            "plots/fig_sensitivity.png": "sensitivity_epsilon.png",
            "plots/fig_sensitivity_full.png": "sensitivity_2x2.png",
        }
        
        # Add mapped files (compressed in memory to avoid 413 Entity Too Large error)
        import io
        from PIL import Image
        for arcname, src_name in file_mapping.items():
            src_path = os.path.join(plots_dir, src_name)
            if os.path.exists(src_path):
                try:
                    with Image.open(src_path) as im:
                        # Resize if width > 800px
                        if im.width > 800:
                            ratio = 800.0 / im.width
                            new_size = (800, int(im.height * ratio))
                            im = im.resize(new_size, Image.Resampling.LANCZOS)
                        
                        # Convert to 8-bit palette to save space
                        im_quantized = im.convert("P", palette=Image.ADAPTIVE, colors=128)
                        
                        # Save in memory
                        img_byte_arr = io.BytesIO()
                        im_quantized.save(img_byte_arr, format="PNG", optimize=True)
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Write to tarball
                        info = tarfile.TarInfo(name=arcname)
                        info.size = len(img_bytes)
                        tar.addfile(info, io.BytesIO(img_bytes))
                        print(f"Added compressed plot: {src_name} as {arcname} ({len(img_bytes)/1024:.1f} KB)")
                except Exception as e:
                    print(f"Error compressing {src_path}: {e}, adding original instead.")
                    tar.add(src_path, arcname=arcname)
            else:
                print(f"Warning: source file {src_path} not found for {arcname}!")
            
    print(f"Tarball created successfully: {tarball_name}")
    
    # 2. Try compiling using latexonline APIs
    # We will try both latexonline.cc and texlive2020.latexonline.cc
    hosts = [
        "https://latexonline.cc",
        "https://latex.aslushnikov.com",
        "https://texlive2020.latexonline.cc"
    ]
    
    success = False
    for host in hosts:
        url = f"{host}/data"
        params = {
            "target": "HO_main2.tex",
            "command": "pdflatex",
            "force": "true"
        }
        
        print(f"\nSending compilation request to {url}...")
        try:
            with open(tarball_name, "rb") as f:
                files = {"file": (os.path.basename(tarball_name), f, "application/x-bzip2")}
                # latexonline expects multipart form data with the file under key 'file'
                response = requests.post(url, params=params, files=files, timeout=120)
                
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                output_pdf = os.path.join(script_dir, "HO_main2.pdf")
                with open(output_pdf, "wb") as pdf_file:
                    pdf_file.write(response.content)
                print(f"Success! Compiled PDF saved as: {output_pdf}")
                success = True
                break
            else:
                print(f"Compilation failed with status code {response.status_code} from {host}.")
                # Print first 1000 characters of error logs if text is returned
                if response.text:
                    print("Error logs / response content snippet:")
                    print(response.text[:1000])
        except Exception as e:
            print(f"Error connecting to {host}: {e}")
            
    # Clean up the tarball
    if os.path.exists(tarball_name):
        os.remove(tarball_name)
        print("\nCleaned up temporary tarball.")
        
    if success:
        print("\nProcess finished successfully!")
    else:
        print("\nFailed to compile PDF via all online compilers.")

if __name__ == "__main__":
    compile_latex()
