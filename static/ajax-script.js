let avatar;
$(document).on('submit', '#animate-form', function (e) {
    console.log('hello');
    e.preventDefault();
    $.ajax({
        type: 'POST',
        url: 'requests',
        data: {
            animate: 1,
            avatar: avatar
        },
        success: function (result) {
        }
    })
});

const buttons = document.querySelectorAll('.button');

buttons.forEach(elem => {

    elem.addEventListener("click", (e) => {
        //removeActive(elem)
        //document.querySelector(".streaming-el:nth-child(2) img").src = elem.childNodes[1].attributes.src.value;
        avatar = elem.childNodes[1].attributes.src.value;
        buttons.forEach(function (e) {
            e.classList.remove('active');
        });
        elem.classList.add('active')
        console.log(avatar)
    })

})

// function removeActive(element) {
//     element = this;
//     if (element.classList.contains('active')) {
//         element.classList.remove('active');
//     } else {
//         buttons.forEach(function (e) {
//             e.classList.remove('active');
//         });
//         element.classList.add('active');
//     }
// }