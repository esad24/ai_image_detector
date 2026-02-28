file_list = ['8bb9230e51fcf2c259f3565cc1b2dc56.jpeg', 'b5c55ed5456a060ceb8ae2b89659506d.jpeg', '8059c316907c586bdf33ad3cb9ca3f95.jpeg', 'daab510ce577421c45ef147f643173f1e7018888813515d82520778e8e789ab1.jpeg', '6e417fd4ac5c9abe2693f1672a22ee58.jpeg', '76377ad07778fc3c355eeb5eb957b8fc.jpeg', 'afa29754521b85a7b8d97210071a1b8a4cf5fafaab6d3cf793a87a81ec55cbfb.jpeg', 'eef7043343ac290520f3f4c0eaced1c5.jpeg', '7f23d59c262d3cfb3eaf43b2ef7d9d6f.jpeg', 'f84a79e62e6e309deacda28d64ded163.jpeg', 'a6d1365c837d5ddb1e24b5f58747cb38.jpeg', 'd3d6d5617d5a71900d9c31fa7bdb6a27.jpeg', '4e243569127164f70aac626c64546d56980e4a85f99f14c2c35807bd787f0cf9.jpeg', '0fdee6e9c98022e1ed3bc440b78b99b3.jpeg', '2aa0e54b4dc251420a3679b501dc8a06.jpeg', 'f4b58700710b3438276f6f52df63310d.jpeg', 'c65519e9d8ee3d4157ae12448e2564ba.jpeg', '77faa1cbee27b073c890bdb2cdc1fa5a.jpeg', 'a094316f760b037374b31f6d3ee5eaa8.jpeg', 'a0ff8ac76a990516504cba6df064ad85.jpeg', 'f7d064ffc492080174abce63c8a3d66c.jpeg', '63f91bafdaa4cc5a093371f96f8c0fa9.jpeg', 'f48d34d25cadb641103890b2b788b389.jpeg', '500c0397385b5a5f3b1ebf434ba34030.jpeg', 'a64f0652261715730b59a150ef80d8ab.jpeg', '55bf29fb606f25f6640805738772d4e04497c0154f8946f458cfc92c26e1cc63.jpeg', '6e7e632843435882e3497ee66c1fb328.jpeg', 'fd23b173438960904e6b21da46e8e0af.jpeg', 'f3158a5cba8b67474f37ea63d3666a32.jpeg', '3c4eb5a796aa88482258435aff4d555f.jpeg', '39db35c3e42f8b591188cffad2160010.jpeg', 'e954a8a31de9ea325f6fc7bc7d0625616c24cf6f22da64264f167702fd8cd0e0.jpeg', 'c83cd234b51b17ff7759c87443a98f49.jpeg', 'def72dae66f159aba9995ef11436ae6184205d851f75cb69691cc815a471bd20.jpeg', '71b124e35753a5993a5f921320e761d7.jpeg', '485db0684c82b3bdd0e828a65f7fa29c05e435591145d1e631413308f8c11d1a.jpeg', 'd22e49191b26998c49b0550b7fdd9cb0.jpeg', 'a062b83843cb1870eb1d66f5591b92cf75f0a1c6523c75ea910cc2fd767ca1b6.jpeg', '427d9398e6271c7b0a4a22f0e3ecb39256cbd96d8179905e5ac1ccf8d4800d22.jpeg', 'ac2a72c405bced22a85f5db5d6bfc7f2d7a98ebc924249ec2e6a5fbb4e82e5ae.jpeg', 'bcd62a29647bf78c402710ba7a9683f9bc8b064187b6643e6ff86fdcc32c2bd2.jpeg', 'ff033f2c6e1d7dc61cb5bc00dfffe267.jpeg', '5f08dee99aede978f2acaefff69e9265.jpeg', '505148b996ff4be5c9690a54ccd2e41a.jpeg', '39cd03d21d22521435aa117fb98e1d3077faf0166ac7e8adbfb7897a2072473f.jpeg', '81e25ad81df43dca638321542c3920ec.jpeg', '539e5299752e361e6a5e79539f7877b201ea0d58f580aa5daab9ea07c0e49405.jpeg', '38139dcf64d5bf28a116249a7709fc4502cc23b21a9b039666f5929621d68778.jpeg', '81d9b444aec14655e05008008aa3a60f.jpeg', '6a98dad4de5ce462fd6c8239be0ddfbc.jpeg', '2ed5f853e9c2b6a92bca20cfcf5a9fe85d353d5ece606d57dba8a024dd1887db.jpeg', '61830a51f7bbafc3d8f17c2e7cce8e10.jpeg', 'dff9b09abad3b7da1e697386be45b8eb.jpeg', 'b580005d0e775d79eea0b051996e34e9.jpeg', '0c9ae7097995139c64b686c6ddec09ba.jpeg', '08a6683aa665b44ca3855432c4f047a40dd029d308091207487deb018f13c872.jpeg', '501f1f9992f910d097c56b82b13e55b00bae418e7c527a1764f2acfd283f39d4.jpeg', '2ad2e05330051435a8976d2f72819b57.jpeg', '80339e93da90ccddc1cbb40b883ec9bc.jpeg', '48e1246a9425640df00bf3390b46d4881e5a6880d2efdb51713f8dac970079c8.jpeg', '9a690d408cee47be9fe65c897f61a477.jpeg', '98c7cae68c2278e3d15448610b0cab61.jpeg', '545b9363e6264074e8cfdd84ccc2a557.jpeg', '79a2a4940acb1def4065e5c15ec37c35.jpeg', 'fdced9599f3452053b4204a0aba2a8adf65bcbb1d529339c1f585423d3c2bf1d.jpeg', '2c4c0b171577884f5c0991cacb5c5ebc.jpeg']


import csv
from collections import Counter


csv_path = "/home/usluesyr/ai_image_detector/data/genClass/images_metadata.csv"

# Dictionary to store filename -> generator
filename_to_generator = {}

# Read CSV
with open(csv_path, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        filename_to_generator[row["filename"]] = row["generator"]

# Count generators only for files in your list
generator_list = []

for filename in file_list:
    if filename in filename_to_generator:
        generator_list.append(filename_to_generator[filename])
    else:
        print(f"Warning: {filename} not found in CSV")

# Count occurrences
generator_counts = Counter(generator_list)

# Print results
print("Generator Counts:")
for generator, count in generator_counts.items():
    print(f"{generator}: {count}")