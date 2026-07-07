import os
import tarfile
import requests

def compile_latex():
    # 1. Create a tarball (.tar.bz2) containing ieee_paper.tex and all plot images
    tarball_name = "latex_project.tar.bz2"
    main_tex = "ieee_paper.tex"
    plots_dir = "plots"
    
    print("Creating tarball...")
    with tarfile.open(tarball_name, "w:bz2") as tar:
        # Add the main tex file
        if os.path.exists(main_tex):
            tar.add(main_tex)
            print(f"Added {main_tex}")
        else:
            print(f"Error: {main_tex} not found!")
            return
        
        # Add all files in the plots directory
        if os.path.exists(plots_dir):
            for file in os.listdir(plots_dir):
                file_path = os.path.join(plots_dir, file)
                if os.path.isfile(file_path):
                    tar.add(file_path)
                    print(f"Added {file_path}")
        else:
            print(f"Warning: {plots_dir} directory not found!")
            
    print(f"Tarball created successfully: {tarball_name}")
    
    # 2. Try compiling using latexonline APIs
    # We will try both latexonline.cc and texlive2020.latexonline.cc
    hosts = [
        "https://latexonline.cc",
        "https://texlive2020.latexonline.cc"
    ]
    
    success = False
    for host in hosts:
        url = f"{host}/data"
        params = {
            "target": main_tex,
            "command": "pdflatex",
            "force": "true"
        }
        
        print(f"\nSending compilation request to {url}...")
        try:
            with open(tarball_name, "rb") as f:
                files = {"file": (tarball_name, f, "application/x-bzip2")}
                # latexonline expects multipart form data with the file under key 'file'
                response = requests.post(url, params=params, files=files, timeout=120)
                
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                output_pdf = "ieee_paper.pdf"
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
